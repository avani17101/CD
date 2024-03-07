from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from re import L
from selectors import EpollSelector
import numpy as np
from utils_tcav2 import *
import torch
from tqdm import tqdm
import os
import torch.nn.functional as F
import torch.nn as nn
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
from statistics import mean
import json
# from train_student_map import *
from tcav_options import *
import math
from argparse import ArgumentParser
import torch.utils.data as tutils
from torch.autograd import Variable
import math
import wandb
import csv
torch.cuda.empty_cache()
from os.path import join as oj
from datetime import datetime
from utils_train import *
from networks_classi import *
import sys
from sklearn.cluster import KMeans
import dep.score_funcs as score_funcs
from torchvision.models import resnet18
from torchvision.models import resnet50
# from DomainGeneralization.Table1.model import MyResnet, MyVGG, MyInceptionResnet, MyAlexnet
# from DomainGeneralization.Table1.PACSDataset import PACSDataset
import torchvision.transforms as transforms
# from train_teacher_map_pacs import DecoMapMNISTNN, EncoMapMNISTNN
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
"""
Copyright 2018 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

print("new script")
opt = TCAVOptions().parse() 

model_type = opt.model_type
dset = opt.dset

device = 'cuda:'+str(0)
### get model, bt, concepts
# model = resnet50(pretrained=True)
kwargs = {'num_workers': 1, 'pin_memory': True}
model_type = "decoymnist"
model, bottleneck_name, train_loader, val_loader, test_loader, X_full, y_full, val_x, val_y, shape = get_model_and_data(model_type, opt, kwargs)
model = model.cuda()
named_layers = dict(model.named_modules())
# bottleneck_name = "layer4.2.conv3"
# shape = (224,224)
# imagenet_zip_path = '/media/Data2/avani.gupta/Imagenet2012/Imagenet-sample/'
# train_dataset = ImageFolder(root=imagenet_zip_path+'val/', transform=val_transforms)
# # train_dataset = ImageFolder(root=imagenet_zip_path+'train/', transform=train_transforms)
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4,
#                                         shuffle=True, num_workers=2)

pairs_vals = "5"                           
num_imgs = 150
pairs_lis = []
for pair_n in pairs_vals.split(","):
    print(model_type,pair_n)
    pairs_lis.append(get_pairs(model_type,concept_set_type=str(pair_n)))

#step 1: calc activations for concepts
bn_activation = None
# for concept in concepts:
def save_activation_hook(mod, inp, out):
    global bn_activation
    bn_activation = out

project_name = '/media/Data2/avani.gupta/IID-Metric/tcav_pt/tcav_class_test_pt_'+dset+'_finetune'

working_dir =  project_name
activation_dir =  working_dir+ '/activations/'
# where CAVs are stored. 
# You can say None if you don't wish to store any.
cav_dir = working_dir + '/cavs/'
cav_hparams = CAV.default_hparams()
cav_hparams['alpha'] = 0.1
cav_hparams['model_type'] = opt.reg_type

tcav_score = {}
cav_directions = {}
acts = {}

for pn,pairs_all in enumerate(pairs_lis):
    
    for main_con in pairs_all:
        pairs = pairs_all[main_con]
        for qs,pair in enumerate(pairs):
            
            concept = pair[0]
            path = '/media/Data2/avani.gupta/imgs_np_'+str(shape[0])+'by'+str(shape[1])+'/'
            if os.path.exists(path+concept+'.npy'):
                imgs_con = np.load(path+concept+'.npy')[:num_imgs]
            else: ##load imgs
                dump_imgs(shape, concept, num_imgs,col_type='rgb')
                imgs_con = np.load(path+concept+'.npy')[:num_imgs]

            acts[concept]= []
            with torch.no_grad():
                model.eval()
                for im in imgs_con:
                    ex = torch.from_numpy(np.moveaxis(im,-1,0)).float()
                    ex.requires_grad = True
                    if opt.model_type == 'toy':
                        ex = ex.reshape(-1,200*200)
                    if model_type == 'decoymnist':
                        ex = ex[0,:,:]
                    handle = named_layers[bottleneck_name].register_forward_hook(save_activation_hook)
                    out = model(ex.unsqueeze(0).cuda())
                    acts[concept].extend(bn_activation.cpu().unsqueeze(0)) 
                    handle.remove()

                if model_type == 'decoymnist':
                    acts[concept] = torch.stack(acts[concept])
                else:
                    acts[concept] = torch.cat(acts[concept])
            
            random_con = pair[1]

            if random_con in acts:
                pass #do nothig
            else:
                path = '/media/Data2/avani.gupta/imgs_np_'+str(shape[0])+'by'+str(shape[1])+'/'
                if os.path.exists(path+random_con+'.npy'):
                    imgs_con = np.load(path+random_con+'.npy')[:num_imgs]
                else: ##load imgs
                    dump_imgs(shape, random_con, num_imgs,col_type='rgb')
                    imgs_con = np.load(path+random_con+'.npy')[:num_imgs]
                
                acts[random_con]= []
                with torch.no_grad():
                    model.eval()
                    for im in imgs_con:
                        ex = torch.from_numpy(np.moveaxis(im,-1,0))
                        ex.requires_grad = True
                        if opt.model_type == 'toy':
                            ex = ex.reshape(-1,200*200)
                        if model_type == 'decoymnist':
                            ex = ex[0,:,:]
                        handle = named_layers[bottleneck_name].register_forward_hook(save_activation_hook)
                        out = model(ex.unsqueeze(0).cuda())
                        acts[random_con].extend(bn_activation.cpu().unsqueeze(0)) 
                        handle.remove()
                    
                    if model_type == 'decoymnist':
                        acts[random_con] = torch.stack(acts[random_con])
                    else:
                        acts[random_con] = torch.cat(acts[random_con])

            #step 3: train cav
            cav = None
            direc_lis = []
            for pair in pairs:
                concept = pair[0]
                affect = 0
                random_con = pair[1]
                if opt.use_nn_cav:
                    cav = get_or_train_cav_nn([concept, random_con],
                                            bottleneck_name,
                                            acts)
                else:
                    cav = get_or_train_cav([concept, random_con],
                                            bottleneck_name,
                                            acts,
                                            cav_dir=cav_dir,
                                            cav_hparams=cav_hparams)

                direc = cav.get_direction(concept) 
                # direc.requires_grad = True
                # print("concept {} negative_concept {} direc {} ".format(concept,random_con,direc))
                direc_lis.append(direc)

            #todo: wrong rn: take mean  cav
            # if take_cav_loss:
            #     if first==0:
            #         cav_loss = torch.sqrt(torch.square(cos(torch.tensor(cav.cavs[0]), torch.tensor(prev_cav.cavs[0])) + cos(torch.tensor(cav.cavs[1]), torch.tensor(prev_cav.cavs[1]))))
            direc_mean = np.mean(np.stack(direc_lis),axis=0) #.detach()
            # direc_mean.requires_grad = True
            direc_mean = torch.tensor(direc_mean)
            if concept in cav_directions: ##update
                cav_directions[concept] = opt.cav_update_wt*direc_mean+(1-opt.cav_update_wt)*cav_directions[concept]

            else:
                cav_directions[concept] = direc_mean

            dot_lis = []
            ### calc tcav scores
            for cur_iter, data in tqdm(enumerate(test_loader)):
                imgs, gt = data
                out = model(imgs.to(device))

                for b in range(imgs.shape[0]):
                    e = imgs[b]
                    def save_activation_hook(mod, inp, out):
                        global bn_activation
                        bn_activation = out
                    handle = named_layers[bottleneck_name].register_forward_hook(save_activation_hook)
                    
                    if model_type == 'toy':
                        out = model(e.unsqueeze(0).to(device).reshape(-1,200*200))
                    else:
                        out = model(e.unsqueeze(0).to(device))

                    act = bn_activation
                    grad_ = None
                    #mapping stuff
                    #teacher mapped to student third last layer
                    # if use_proto:
                    #     # if use_knn_proto:
                    #     for f in range(knn_k):
                    #         if f==0:
                    #             loss = mse_loss(act, torch.from_numpy(proto_dic[gt[b].item()]['cluster_'+str(f)]).cuda().unsqueeze(0))
                    #         else:
                    #             loss += mse_loss(act, torch.from_numpy(proto_dic[gt[b].item()]['cluster_'+str(f)]).cuda().unsqueeze(0))
                    #     # else:  
                    #         # loss = mse_loss(act, torch.from_numpy(proto_dic[gt[b].item()]).cuda()) #loss btw act and proto
                    #     grad_ = torch.autograd.grad(loss, act, retain_graph=True,create_graph=True)
                        
                    # else:
                    grad_ = torch.autograd.grad(out[0][gt[b]], act, retain_graph=True,create_graph=True)
                    
                    dot = torch.dot(grad_[0].to(device).float().squeeze(0).flatten(),direc_mean.to(device).float())/torch.linalg.norm(direc_mean)
                    dot_lis.append(dot)
                    

            dot_lis = torch.stack(dot_lis)
            tcav_score[concept] = len(torch.where(dot_lis<0)[0])/len(dot_lis)
        
        
print(tcav_score)
print("avg", np.mean(list(tcav_score.values())))
