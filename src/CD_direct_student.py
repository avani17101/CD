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
from tcav_options_direct_student import *
import math
from tqdm import tqdm
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
import glob

# from DomainGeneralization.Table1.model import MyResnet, MyVGG, MyInceptionResnet, MyAlexnet
# from DomainGeneralization.Table1.PACSDataset import PACSDataset
import torchvision.transforms as transforms
# from train_teacher_map_pacs import DecoMapMNISTNN, EncoMapMNISTNN

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
print("current seed",torch.seed())
cur_seed = 9612836376807465428
torch.manual_seed(cur_seed)
torch.cuda.manual_seed(cur_seed)
# opt.model_type = 'colormnist'
# opt.dset = 'mnist'
print(opt)
wandb.init(project="change_cav_exp_"+opt.model_type, entity="avani", config=opt)
# wandb.init(config=opt, save_code=True, resume= True)

model_type = opt.model_type
dset = opt.dset
affect = 0
epochs = 50
affect_lis = [0]
lr = 1e-4
barlow_lamb = opt.barlow_lamb
wtcav = 10e-10
cdep_grad_method = opt.cdep_grad_method
regularizer_rate = opt.regularizer_rate
freeze_bef_layers = opt.freeze_bef_layers
pairs_vals =str(opt.pairs_vals)

batch_size = opt.batch_size
knn_k = opt.knn_k
cur_proto_mean_wt = opt.cur_proto_mean_wt
use_proto = opt.use_proto
use_last_layer_proto = opt.use_last_layer_proto
use_cdepcolor = opt.use_cdepcolor
use_precalc_proto = opt.use_precalc_proto
update_proto = opt.update_proto
train_from_scratch = opt.train_from_scratch
use_knn_proto= opt.use_knn_proto
class_wise_training = opt.class_wise_training
num_imgs = opt.num_imgs
bias = "1"
print(opt.gpu_ids)
use_cuda = torch.cuda.is_available()

from imdb_classi_test import *
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
if model_type == 'faces':
    shape = (224,224)
    model = resnet18(pretrained=False)
    model.fc = nn.Linear(512,2)
    if not opt.train_from_scratch:
        model.load_state_dict(torch.load("/media/Data2/avani.gupta/new_checkpoints/acc59.0best_val_acc99.5highest_epoch0iter0facesfacesp8nimg150lr0.01rr0.3wtcav5bs44pwt0.3upr1cd0precalc1up1scratch0uknn1cwt0s42corr"))
    model.cuda()
    model.eval()
    bs = 64
    train_loader = DataLoader(bFFHQDataset("train"),bs, shuffle=True, num_workers=0)
    val_loader = DataLoader(bFFHQDataset("valid"),bs, shuffle=True, num_workers=0)
    test_loader = DataLoader(bFFHQDataset("test"),bs, shuffle=True, num_workers=0)
    named_layers = dict(model.named_modules())
    lis = list(named_layers.keys())
    bottleneck_name = 'layer4.1.conv1'

if model_type =='cat_dog':
    model = models.resnet18(pretrained=True)
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, 1)
    bias = 'TB'+str(bias)
    model.load_state_dict(torch.load('/home/avani/tcav_pt/cat_dog_model'+bias+'.pt'))
    model = model.cuda()
    model.eval()

    bs = 64
    if bias=='TB2':
        dset = 'cat_dog'+bias
    dset ='cat_dog' #+bias
    
    # dset = 'imdb'
    # bias = 'EB2' #[TB1, TB2 for cat_dot], EB1, EB2 for imdb
    data_transform = transforms.Compose([transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
    print("catt")
    train_ds = CatBiasedDataSet(data_transform,bias,mode='train')
    train_loader = DataLoader(train_ds, bs, shuffle=True, num_workers=3)
    val_ds = CatBiasedDataSet(data_transform,bias,mode='val')
    val_loader = DataLoader(val_ds, bs, shuffle=True, num_workers=3)

    if bias == "1":
        opp = "2"
    else:
        opp = "1"
    if dset =='cat_dog':
        bias = "TB"+bias
        opp_bias = "TB"+opp
    else:
        bias = "EB"+bias
        opp_bias = "EB"+opp
    print("opp_bias",opp_bias)
    ds = CatBiasedDataSet(data_transform,opp_bias,mode='test')  #use entire set of opposite set
    test_loader = DataLoader(ds, bs)

    named_layers = dict(model.named_modules())
    lis = list(named_layers.keys())
    bottleneck_name = 'layer4.1.conv1'


if model_type == 'decoymnist':
    dpath = 'dep/data/DecoyMNIST/'
    X_full = torch.Tensor(np.load(oj(dpath, "train_x_decoy.npy")))
    y_full = torch.Tensor(np.load(oj(dpath, "train_y.npy"))).type(torch.int64)
    complete_dataset = tutils.TensorDataset(X_full, y_full) # create your datset

    num_train = int(len(complete_dataset)*.9)
    num_test = len(complete_dataset)  - num_train 
    torch.manual_seed(0)
    train_dataset, val_dataset,= torch.utils.data.random_split(complete_dataset, [num_train, num_test])
    train_loader = tutils.DataLoader(train_dataset, batch_size=num_imgs, shuffle=True, **kwargs) # create your dataloader
    val_loader = tutils.DataLoader(val_dataset, batch_size=num_imgs, shuffle=True, **kwargs) # create your dataloader

    test_x_tensor = torch.Tensor(np.load(oj(dpath, "test_x_decoy.npy")))
    test_y_tensor = torch.Tensor(np.load(oj(dpath, "test_y.npy"))).type(torch.int64)
    test_dataset = tutils.TensorDataset(test_x_tensor,test_y_tensor) # create your datset
    test_loader = tutils.DataLoader(test_dataset, batch_size=num_imgs, shuffle=True, **kwargs) # create your dataloader
    
    model = MNISTDecoyNet()
    model.cuda()
    shape = (28,28)
    if not train_from_scratch:
        model.load_state_dict(torch.load('mnist/DecoyMNIST/orig_model_decoyMNIST_.pt'))
    bottleneck_name = 'conv2'

if model_type =='colormnist':
    x_numpy_train = np.load(os.path.join("dep/data/ColorMNIST", "train_x.npy"))
    prob = (x_numpy_train.sum(axis = 1) > 0.0).mean(axis = 0).reshape(-1)
    prob /=prob.sum()
    mean = x_numpy_train.mean(axis = (0,2,3))
    std = x_numpy_train.std(axis = (0,2,3))
    bottleneck_name = opt.bottleneck_name
    if bottleneck_name=='conv1':
        dset = dset + 'conv1'

    def load_dataset(name, path='dep/'):
        x_numpy = np.load(os.path.join(path+"data/ColorMNIST", name + "_x.npy"))
        x_numpy -= mean[None, :, None, None,]
        x_numpy /= std[None, :, None, None,]
        y_numpy = np.load(os.path.join(path+"data/ColorMNIST", name +"_y.npy"))
        x_tensor = torch.Tensor(x_numpy)
        y_tensor = torch.Tensor(y_numpy).type(torch.int64)
        dataset = tutils.TensorDataset(x_tensor,y_tensor) 
        return dataset, x_tensor, y_tensor

    train_dataset,X_full,y_full = load_dataset("train")
    val_dataset,val_x, val_y  = load_dataset("val")
    test_dataset,_,_ = load_dataset("test")
    train_loader = tutils.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    val_loader = tutils.DataLoader(val_dataset,batch_size=batch_size)
    test_loader = tutils.DataLoader(test_dataset,batch_size=batch_size)
    model = MNISTColorNet()
    model.cuda()
    shape = (28,28)

    if not train_from_scratch:
        # model.load_state_dict(torch.load('/media/Data2/avani.gupta/new_checkpoints/acc46.64best_val_acc45.7highest_epoch0iter130colormnistcolormnistp8nimg150lr0.01rr0.3wtcav5bs44pwt0.3upr1cd0precalc1up1scratch0uknn1cwt0s42corr'))
        model.load_state_dict(torch.load('mnist/ColorMNIST/orig_model_colorMNIST.pt'))

if model_type =='texturemnist':
    x_numpy_train = np.load(os.path.join("/media/Data2/avani.gupta/data/ColorMNIST", "train_texture_x.npy"))
    prob = (x_numpy_train.sum(axis = 1) > 0.0).mean(axis = 0).reshape(-1)
    prob /=prob.sum()
    mean = x_numpy_train.mean(axis = (0,2,3))
    std = x_numpy_train.std(axis = (0,2,3))
    bottleneck_name = 'conv2'
    if bottleneck_name=='conv1':
        dset = dset + 'conv1'

    def load_dataset(name, path='dep/'):
        x_numpy = np.load(os.path.join(path+"data/ColorMNIST", name + "_x.npy"))
        x_numpy -= mean[None, :, None, None,]
        x_numpy /= std[None, :, None, None,]
        y_numpy = np.load(os.path.join(path+"data/ColorMNIST", name +"_y.npy"))
        x_tensor = torch.Tensor(x_numpy)
        y_tensor = torch.Tensor(y_numpy).type(torch.int64)
        dataset = tutils.TensorDataset(x_tensor,y_tensor) 
        return dataset, x_tensor, y_tensor

    train_dataset,X_full,y_full = load_dataset("train_texture")
    val_dataset,val_x, val_y  = load_dataset("val_texture")
    test_dataset,_,_ = load_dataset("test_texture")
    train_loader = tutils.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    val_loader = tutils.DataLoader(val_dataset,batch_size=batch_size)
    test_loader = tutils.DataLoader(test_dataset,batch_size=batch_size)
    model = MNISTColorNet()
    model.cuda()
    shape = (28,28)
    if not train_from_scratch:
        model.load_state_dict(torch.load('/home/avani/tcav_pt/main_files/methodvanilla_tex.pt'))

s = ""
if train_from_scratch:
    s = "scratch_train"
else:
    s = "finetune"

root = "/media/Data2/avani.gupta"
full_root = root +''
device = 'cuda:'+str(0)

test_code = opt.test_code
finetune = opt.finetune
get_tcav_score = opt.get_tcav_score
num_random_exp = opt.num_random_exp
use_triplet_loss = opt.use_triplet_loss

if test_code:
    exp_name = model_type+"test"+bottleneck_name+"loss_"+opt.loss_type
else:
    exp_name = model_type+bottleneck_name+"loss_"+opt.loss_type

def srgb_to_rgb(srgb):
    ret = torch.zeros_like(srgb)
    idx0 = srgb <= 0.04045
    idx1 = srgb > 0.04045
    ret[idx0] = srgb[idx0] / 12.92
    ret[idx1] = torch.pow((srgb[idx1] + 0.055) / 1.055, 2.4)
    return ret

def rgb_to_chromaticity(rgb):
    """ converts rgb to chromaticity """
    irg = torch.zeros_like(rgb)
    s = torch.sum(rgb, axis=-1) + 1e-6
    irg[..., 0] = rgb[..., 0] / s
    irg[..., 1] = rgb[..., 1] / s
    irg[..., 2] = rgb[..., 2] / s
    return irg

def save_activation_hook(mod, inp, out):
    global bn_activation
    bn_activation = out

pairs_lis = []
# affect_lis = []
for pair_n in pairs_vals.split(","):
    pairs_lis.append(get_pairs(model_type,concept_set_type=str(pair_n)))
    # proto_dic = np.load('/media/Data2/avani.gupta/proto_'+model_type+"knn"+str(knn_k)+'mnistconv1.npy',allow_pickle=True)[()] #load proto dicts
    # else:
    #     proto_dic = np.load('/media/Data2/avani.gupta/proto_'+model_type+"knn"+str(knn_k)+'.npy',allow_pickle=True)[()] #load proto dicts


print(pairs_lis)

named_layers = dict(model.named_modules())
lis = list(named_layers.keys())
model.train()

#concept loss params
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
results1 = []
alpha = 0.1
direc_mean = None
sig_data = None
alpha = 0.1
relu = nn.ReLU()
loss_type = 'L1_cos'
mse_loss = nn.MSELoss()
result_end = []
# imgs, gt = get_eval_set_arap()
tcav_loss = None
cav_loss = None
cav_directions = {'R':{}, "S":{}}
acts = {}
concepts = []
z = None
whdr = None
bn_activation = None
cos = nn.CosineSimilarity(dim=0,eps=1e-6)
mse = nn.MSELoss()
criterion = nn.CrossEntropyLoss()
# criterion2 = nn.CrossEntropyLoss(reduction='none')
# criterion_ = nn.CrossEntropyLoss()
criterion_bce = nn.BCEWithLogitsLoss()
triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=nn.CosineSimilarity(dim=0,eps=1e-6))
if model_type =='clever_hans':
    class_labels = [0,1,2] #for lb 0 gray concept should not affect, for lb 1 metal should not
elif model_type =='clever_hans7':
    class_labels = [0,1,2,3,4,5,6,7]
elif model_type =='faces':
    class_labels = [0,1]

start_ep = opt.concept_train_start_ep
def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

if opt.finetune:
    start_ep = 0

model_save_name = model_type+dset
k = ["pairs_vals","num_imgs","lr","regularizer_rate","wtcav","batch_size","cur_proto_mean_wt","use_proto","use_cdepcolor","use_precalc_proto","update_proto","train_from_scratch","use_knn_proto","class_wise_training"]
ks = {"pairs_vals":"p","num_imgs":"nimg","lr":"lr","regularizer_rate":"rr","wtcav":"wtcav", "batch_size":"bs","cur_proto_mean_wt":"pwt","use_proto":"upr","use_cdepcolor":"cd","use_precalc_proto":"precalc","update_proto":"up","train_from_scratch":"scratch","use_knn_proto":"uknn","class_wise_training":"cwt"}
dic = opt.__dict__
for p in k:
    if isinstance(dic[p],str):
        model_save_name += ks[p]+str(dic[p])
    elif isinstance(dic[p],(int, float)):
        model_save_name += ks[p]+str(round(dic[p],2))
    else:
        model_save_name += ks[p]+str(dic[p])
model_save_name += "s"+str(opt.seed)
print("round saving models ",model_save_name)
    
res = {}
import torch.optim as optim
optimizer = optim.Adam(model.parameters(), lr=1e-4)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
#     factor=0.1, patience=10, threshold=0.0001, threshold_mode='abs')

if model_type in ['toy','colormnist','decoymnist',"texturemnist"]:
    class_labels = y_full.unique().numpy()
    # for c in class_labels:
    #     idx, = torch.where(y_full==c)
    #     lb_wise_X[c] = X_full[idx] 
num_iterations_in_one_epoch = len(train_loader.dataset)
print("num_iterations_in_one_epoch ",num_iterations_in_one_epoch)
cos = nn.CosineSimilarity(dim=0,eps=1e-6)
proto_dic = {}
best_val_acc = -1
corr_test_acc = 0
for epoch in tqdm(range(epochs)):
    '''
    if cav freq or first epoch calc cav
    '''
    triplet_loss_lis = []
    cav_loss_lis = []
    if use_proto:
        if epoch==0 and use_precalc_proto:
            if opt.dset == 'mnistconv1':
                proto_dic = np.load('/media/Data2/avani.gupta/proto_colormnistconv1knn7mnistconv1.npy',allow_pickle=True)[()] #load proto dicts
            else:
                proto_dic = np.load('/media/Data2/avani.gupta/proto_'+model_type+"knn"+str(knn_k)+'.npy',allow_pickle=True)[()] #load proto dicts
     
        elif epoch%2==0 and update_proto:
            max_classwise_samples = 1000 ##max class-samples for proto-type creation
            # if 'clever_hans' in model_type:  ###use in case dataset too big (need proto-type loader to be passed as well)
            #     data= next(iter(train_proto_loader))
            #     X_full, _, y_full, _, _, _ = data
            #     del data
            #     bss = 50

            for c in class_labels:
                acts = []
                def save_activation_hook(mod, inp, out):
                    global bn_activation
                    bn_activation = out
                idx, = torch.where(y_full==c)
                if len(idx)>0:
                    handle = named_layers[bottleneck_name].register_forward_hook(save_activation_hook)
                    for i in range(0,len(X_full[idx]),max_classwise_samples):
                        if model_type == 'toy':
                            _ = model(X_full[idx][i:min(i+max_classwise_samples,len(X_full[idx]))].cuda().reshape(-1,200*200))
                        else:
                            _ = model(X_full[idx][i:min(i+max_classwise_samples,len(X_full[idx]))].cuda())

                        acts.append(bn_activation)
                    acts = torch.concat(acts,axis=0)
                    ss = acts.shape

                    #giving half weightage to old proto mean and half to new proto
                    # if use_knn_proto:
                    acts = acts.detach().cpu().numpy().reshape((-1,ss[1]*ss[2]*ss[3]))
                    kmeans = KMeans(n_clusters=knn_k, random_state=0).fit(acts)
                    centers = kmeans.cluster_centers_
                    for f in range(knn_k):
                        if epoch==0 or not opt.do_proto_mean:
                            if c in proto_dic:
                                proto_dic[c]['cluster_'+str(f)] = centers[f].reshape(ss[1],ss[2],ss[3])
                            else:
                                proto_dic[c] = {}
                                proto_dic[c]['cluster_'+str(f)] = centers[f].reshape(ss[1],ss[2],ss[3])
                        else:
                            proto_dic[c]['cluster_'+str(f)] = (1-cur_proto_mean_wt)*proto_dic[c]['cluster_'+str(f)]+cur_proto_mean_wt*centers[f].reshape((ss[1],ss[2],ss[3]))

    #proto acts as psuedo GT
    #train only over last k layers
    # or train only bottleneck
    enco_layer = bottleneck_name
    lis = list(named_layers.keys())
    print("freezing following: ")
    if freeze_bef_layers:# and not train_from_scratch:
        idx_bt = lis.index(enco_layer) #freeze layers untill here
        for i in range(idx_bt+1):
            if lis[i] != '':
                print(lis[i])
                for p in named_layers[lis[i]].parameters():
                    p.requires_grad = False
    
    def are_parameters_trainable(layer):
        for param in layer.parameters():
            if not param.requires_grad:
                return False
        return True
    
    for i in range(len(lis)):
        print(lis[i], are_parameters_trainable(named_layers[lis[i]]))
        
    cur_meh = 0

    ##actual training iterations: over train_dataset
    for cur_iter, data in enumerate(train_loader):
        if model_type =='clever_hans' or model_type=='clever_hans7':
            imgs, _, gt, _, _, _ = data
        else:
            imgs, gt = data
        if model_type=='faces': #reduce one hot encoding to binry (for faces its just binary classi so take first col as labels)
            gt = gt[:,0]

        #calc gt loss
        imgs.requires_grad = True

        if model_type == 'toy':
            out = model(imgs.to(device).reshape(-1,200*200))
        else:
            out = model(imgs.to(device))
        
        ## check which training gt loss used by base model (check softmax etc well)
        if model_type == 'colormnist' or model_type =='decoymnist' or model_type=='texturemnist':
            gt_loss = torch.nn.functional.nll_loss(out, gt.to(device))
        elif model_type =='faces':
            gt_loss = criterion(out, gt.to(device))
        elif model_type == 'cat_dog':
            gt_loss= criterion_bce(out, gt.view(len(gt), 1).to(device))
        else:
            gt_loss = criterion(out,gt.to(device))

        ##local cdep loss
        if use_cdepcolor:
            if model_type =='colormnist' or model_type=='texturemnist' or model_type =='cat_dog' or model_type =='faces':
                num_blobs = 8
                regularizer_rate = regularizer_rate
                blobs = np.zeros((28*28,28,28))
                for i in range(28):
                    for j in range(28):
                        blobs[i*28+j, i, j] =1
                imgs = imgs.cuda()
                add_loss = torch.zeros(1,).cuda()
                blob_idxs = np.random.choice(28*28, size = num_blobs, p = prob)
                if cdep_grad_method ==0:
                    for i in range(num_blobs): 
                        add_loss += score_funcs.cdep(model, imgs, blobs[blob_idxs[i]],model_type = 'mnist')
                    (regularizer_rate*add_loss+gt_loss).backward(retain_graph=True)
                elif cdep_grad_method ==1:
                    for i in range(num_blobs): 
                        add_loss +=score_funcs.gradient_sum(imgs, gt.to(device), torch.FloatTensor(blobs[blob_idxs[i]]).to(device),  model, torch.nn.functional.nll_loss)
                    (regularizer_rate*add_loss).backward(retain_graph=True)
                    # loss = torch.nn.functional.nll_loss(out, gt.to(device))
                    # loss.backward(retain_graph=True)
                elif cdep_grad_method ==2:
                    for j in range(len(imgs)):
                        for i in range(num_blobs): 
                            add_loss +=(score_funcs.eg_scores_2d(model, imgs, j, gt.to(device), 50) * torch.FloatTensor(blobs[blob_idxs[i]]).to(device)).sum()

                    (regularizer_rate*add_loss).backward(retain_graph=True)
                # loss = torch.nn.functional.nll_loss(out, gt.to(device))
                # loss.backward(retain_graph=True)
            elif model_type =='decoymnist': ##for grayscale networks
                regularizer_rate = regularizer_rate
                import dep.cdep as cdep
                num_blobs = 1
                num_samples =200
                blob = np.zeros((28,28))
                size_blob =5
                imgs = imgs.cuda()

                blob[:size_blob, :size_blob ] =1
                blob[-size_blob:, :size_blob] = 1
                blob[:size_blob, -size_blob: ] =1
                blob[-size_blob:, -size_blob:] = 1
                if regularizer_rate !=0:
                    add_loss = torch.zeros(1,).cuda()
                    if cdep_grad_method ==0:
                        rel, irrel = cdep.cd(blob, imgs,model)
                        add_loss += torch.nn.functional.softmax(torch.stack((rel.view(-1),irrel.view(-1)), dim =1), dim = 1)[:,0].mean()
                        (regularizer_rate*add_loss +gt_loss).backward(retain_graph=True)
                    elif cdep_grad_method ==1:
                        add_loss +=score_funcs.gradient_sum(imgs, gt.to(device), torch.FloatTensor(blob).to(device),  model, torch.nn.functional.nll_loss)
                        (regularizer_rate*add_loss).backward(retain_graph=True)

                    elif cdep_grad_method ==2:
                        for j in range(len(imgs)):
                            add_loss +=(score_funcs.eg_scores_2d(model, imgs, j, gt.to(device), num_samples) * torch.FloatTensor(blob).to(device)).sum()
                        (regularizer_rate*add_loss).backward(retain_graph=True)

        if opt.use_gt_loss: ##if use gt loss 
            gt_loss.backward(retain_graph=True)
        
        ## iterate over concept pairs (pos vs neg concepts and aggregate concept loss)
        faces_class_wise = {0:"young_women",1:"old_men"}
        for pn,pairs_all in enumerate(pairs_lis):
            for positive_con in pairs_all: 
                pairs = pairs_all[positive_con]
                for qs,pair in enumerate(pairs): ##aggregative loss over all pairs of positive_con (positive_con vs all neg_con )
                    affect = affect
                    concept = pair[0]
                    new_cav_freq = opt.cav_update_freq
                    if opt.set_to_ep_level:
                        new_cav_freq = num_iterations_in_one_epoch
                    
                    if (cur_iter % new_cav_freq  == 0 and opt.update_cavs) or (cur_iter==0 and epoch==0): #train new cav's
                        #step 1: calc activations for concepts
                        bn_activation = None
                        def save_activation_hook(mod, inp, out):
                            global bn_activation
                            bn_activation = out
                        
                        path = '/media/Data2/avani.gupta/imgs_np_'+str(shape[0])+'by'+str(shape[1])+'/'
                        if cur_iter == 0 and epoch ==0:
                            imgs_con = np.load(path+concept+'.npy')[:num_imgs]
                        else:
                            imgs_con = np.load(path+concept+'.npy')[num_imgs:num_imgs+cur_meh]

                        acts[concept]= []
                        with torch.no_grad():
                            model.eval()
                            for im in imgs_con:
                                ex = torch.from_numpy(np.moveaxis(im,-1,0)).float()
                                ex.requires_grad = True
                                if opt.model_type == 'toy':
                                    ex = ex.reshape(-1,200*200)
                                if opt.model_type == 'decoymnist':
                                    ex = ex[0,:,:]
                                handle = named_layers[bottleneck_name].register_forward_hook(save_activation_hook)
                                out = model(ex.unsqueeze(0).cuda())
                                acts[concept].extend(bn_activation.cpu().unsqueeze(0)) 
                                handle.remove()
                            if opt.model_type=='decoymnist':
                                acts[concept] = torch.stack(acts[concept])
                            else:
                                acts[concept] = torch.cat(acts[concept])
                        
                        random_con = pair[1]
                        if random_con in acts:
                            pass #do nothig
                        else:
                            path = '/media/Data2/avani.gupta/imgs_np_'+str(shape[0])+'by'+str(shape[1])+'/'
                            imgs_con = np.load(path+random_con+'.npy')[:num_imgs]
                            acts[random_con]= []
                            with torch.no_grad():
                                model.eval()
                                for im in imgs_con:
                                    ex = torch.from_numpy(np.moveaxis(im,-1,0))
                                    ex.requires_grad = True
                                    if opt.model_type == 'toy':
                                        ex = ex.reshape(-1,200*200)
                                    if opt.model_type == 'decoymnist':
                                        ex = ex[0,:,:]
                                    handle = named_layers[bottleneck_name].register_forward_hook(save_activation_hook)
                                    out = model(ex.unsqueeze(0).cuda())
                                    acts[random_con].extend(bn_activation.cpu().unsqueeze(0)) 
                                    handle.remove()
                                if opt.model_type=='decoymnist':
                                    acts[random_con] = torch.stack(acts[random_con])
                                else:
                                    acts[random_con] = torch.cat(acts[random_con])

                        #step 3: train cav
                        cav = None
                        direc_lis = []
                        for pair in pairs:
                            concept = pair[0]
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
                            direc_lis.append(direc)

                        #todo: wrong rn: take mean  cav
                        # if take_cav_loss:
                        #     if first==0:
                        #         cav_loss = torch.sqrt(torch.square(cos(torch.tensor(cav.cavs[0]), torch.tensor(prev_cav.cavs[0])) + cos(torch.tensor(cav.cavs[1]), torch.tensor(prev_cav.cavs[1]))))
                        
                        direc_mean = np.mean(np.stack(direc_lis),axis=0) ##take mean of positive con vs all negative counterpart directions
                        # direc_mean.requires_grad = True
                        direc_mean = torch.tensor(direc_mean)
                        if concept in cav_directions: ##update
                            cav_directions[concept] = opt.cav_update_wt*direc_mean+(1-opt.cav_update_wt)*cav_directions[concept]

                        else:
                            cav_directions[concept] = direc_mean
                        # del cav
                        acts[concept] = {}
                    
                    else: #use older mean cav directions
                        direc_mean = cav_directions[concept]
                        if acts:
                            acts = {}

                    if concept=='women' or concept=='men':
                        affect = 1 #men vs women concept shall affect
                    
                    named_layers = dict(model.named_modules())
                    bn_activation= None
                    grad = None
                    tcav_score = {}
                    results = []
                    ep_loss = 0
                    dot_lis = []
                    cos_lis = []
                    mse_dvec_lis = []
                    grad_lis  =[]
                    tcav_score_lis = []
                    concept_loss_lis = []

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
                        if use_proto:
                            # if use_knn_proto:                                
                            for f in range(knn_k):
                                if f==0:
                                    loss = mse_loss(act, torch.from_numpy(proto_dic[gt[b].item()]['cluster_'+str(f)]).cuda().unsqueeze(0))
                                else:
                                    loss += mse_loss(act, torch.from_numpy(proto_dic[gt[b].item()]['cluster_'+str(f)]).cuda().unsqueeze(0))
                            # else:  
                                # loss = mse_loss(act, torch.from_numpy(proto_dic[gt[b].item()]).cuda()) #loss btw act and proto
                            grad_ = torch.autograd.grad(loss, act, retain_graph=True,create_graph=True)
                            
                        else:
                            if use_last_layer_proto: ##proto in last layer
                                for f in range(knn_k):
                                    if f==0:
                                        loss = mse_loss(out[0][gt[b]], torch.from_numpy(proto_dic[gt[b].item()]['cluster_'+str(f)]).cuda().unsqueeze(0))
                                    else:
                                        loss += mse_loss(out[0][gt[b]], torch.from_numpy(proto_dic[gt[b].item()]['cluster_'+str(f)]).cuda().unsqueeze(0))
                                grad_ = torch.autograd.grad(loss, act, retain_graph=True,create_graph=True)
                            else: #Logit
                                grad_ = torch.autograd.grad(out[0][gt[b]], act, retain_graph=True,create_graph=True)
                        # loss = criterion(out,gt[b].unsqueeze(0).to(device))
                        # if loss_type=='mse':
                        if 'cos' in loss_type:
                            cos_ = cos(grad_[0].to(device).float().squeeze(0).flatten(),direc_mean.to(device).float())
                            cos_lis.append(cos_)
                        if 'barlow' in loss_type:
                            grad_lis.append(grad_[0])
                        else:
                            dot = torch.dot(grad_[0].to(device).float().squeeze(0).flatten(),direc_mean.to(device).float())/torch.linalg.norm(direc_mean)
                            dot_lis.append(dot)
                            unit_grad = grad_[0].to(device).float().squeeze(0).flatten()
                            unit_direc = direc_mean.to(device).float()/torch.linalg.norm(direc_mean)
                            unit_grad = unit_grad/torch.linalg.norm(unit_grad)
                            mse_dvec_lis.append(mse_loss(unit_grad, unit_direc))
                        handle.remove()

                    if 'barlow' in loss_type:
                        grad_all = torch.stack(grad_lis).unsqueeze(1)
                        grad_all = grad_all.reshape(grad_all.shape[0],-1)
                        bn = nn.BatchNorm1d(grad_all.shape[1], affine=False).cuda()
                        c = bn(grad_all).T @ bn(direc_mean.repeat(grad_all.shape[0],1).cuda().float())
                        # c = bn(grad_all).T @ bn(direc_mean)
                        if affect==1:
                            on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
                        else:
                            on_diag = torch.diagonal(c).pow_(2).sum()
                        off_diag = off_diagonal(c).pow_(2).sum()
                        loss_ = on_diag + barlow_lamb * off_diag
                        loss_ = wtcav*loss_
                    else:
                        dot_lis = torch.stack(dot_lis)
                        mse_dvec_lis = torch.stack(mse_dvec_lis)
                        if 'cos' in loss_type:
                            cos_lis = torch.stack(cos_lis)

                        #or cs(grad and direc_mean) max?
                        score = len(torch.where(dot_lis<0)[0])/len(dot_lis)
                        tcav_score_lis.append(score)
                        if loss_type =='cos':
                            if affect==False:
                                loss_ = -wtcav*torch.sum(cos_lis) 
                        
                        if affect==True:
                            # if loss_type =='cos_mse':  
                            loss_= mse_loss(cos_lis, torch.ones(len(cos_lis)).to(device))
                                # loss_ = torch.sum(torch.abs(cos_lis)) #L1  direct hence cos 0 = 1 | affect | vectors aligned

                        if loss_type =='L1_cos':
                            if affect==False:
                                loss_ = torch.sum(torch.abs(cos_lis)) #L1

                        if loss_type =='L2_cos':
                            if affect==False:
                                loss_ = torch.sum(torch.square(cos_lis)) #L2
                        
                        if loss_type =='mse_vecs':
                            if affect==False:
                                loss_ = torch.sum(mse_dvec_lis)

                        if loss_type=='mse':
                            dot_lis_normalised = (dot_lis)/torch.max(dot_lis)
                            if affect!=None:
                                if affect==False:
                                    # print("hereeee",max(dot_lis), mean(dot_lis), min(dot_lis))
                                    dot_lis = torch.reshape(dot_lis, (-1,))
                                    target = torch.ones(len(dot_lis)).to(device)
                                    tcav_loss = None
                                    if model_type=='cg':
                                        tcav_loss = mse_loss(dot_lis.unsqueeze(0).to(device).double(), target.unsqueeze(0).to(device).double())
                                    elif model_type=='toy':
                                        tcav_loss = criterion_(dot_lis.unsqueeze(0).to(device), target.unsqueeze(0).to(device))
                                    elif model_type=='colormnist':
                                        tcav_loss = criterion_(dot_lis.unsqueeze(0).to(device), target.unsqueeze(0).to(device))
                                    else:
                                        tcav_loss = criterion_(dot_lis.unsqueeze(0).to(device), target.unsqueeze(0).to(device))
                                    loss_ = wtcav*tcav_loss
                    
                    cav_loss_lis.append(loss_.item())
                    if qs==len(pairs):
                        loss_.backward()
                    else:
                        loss_.backward(retain_graph=True)
                
        optimizer.step()
        optimizer.zero_grad()
        model.eval()
      
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                if model_type=='faces': #reduce one hot encoding to binry (for faces its just binary classi so take first col as labels)
                    target = target[:,0]
                output = model(data)
                if model_type == 'faces':
                    val_loss += criterion(output, target).item()
                elif model_type=='cat_dog':
                    val_loss += criterion_bce(output, target.view(len(target), 1).to(device)).item()
                else:
                    val_loss += torch.nn.functional.nll_loss(output, target, reduction='sum').item() # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()


        val_loss /= len(val_loader.dataset)
        val_acc = 100. * correct / len(val_loader.dataset)


        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                if model_type=='faces': #reduce one hot encoding to binry (for faces its just binary classi so take first col as labels)
                    target = target[:,0]
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_acc = 100. * correct / len(test_loader.dataset)
        if val_acc > best_val_acc:
            wandb.run.summary[f"best_val_acc_"] = val_acc
            wandb.run.summary[f"best_val_corr_test_acc"] = test_acc
            wandb.run.summary[f"best_val_epoch"] = epoch
            wandb.run.summary[f"best_val_iter"] = cur_iter
            wandb.run.summary[f"best_val_acc_cpk"] = model_save_name+"corr"
            torch.save(model.state_dict(), os.path.join(opt.checkpoints_dir,"acc"+str(round(test_acc,4))+"best_val_acc"+str(round(val_acc,2))+"highest_epoch"+str(epoch)+"iter"+str(cur_iter)+model_save_name+"corr"))
            best_val_acc = val_acc
            corr_test_acc = test_acc
        model.train()

        wandb.log({"score": tcav_score_lis, "gt_loss": gt_loss, "cav_loss":sum(cav_loss_lis) ,"val_gt_loss":val_loss, "val_acc":val_acc, "test_gt_loss":test_loss,"test_acc":test_acc,"best_val_acc":best_val_acc, "corr_test_acc":corr_test_acc})#,"small_num_op_loss":small_num_op_loss_mean})
        cav_loss_lis = []

print("we are done!!!!!")