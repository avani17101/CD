from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from re import L
from selectors import EpollSelector
import numpy as np
from utils.utils_tcav2 import CAV
from utils.utils_tcav2 import *
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
from train_student_map import *
from utils.tcav_options import *
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
from utils.utils_train import *
from utils.networks_classi import *
import sys
from sklearn.cluster import KMeans
import dep.score_funcs as score_funcs
from torchvision.models import resnet18
from torchvision.models import resnet50
from torchvision.datasets import ImageFolder


# from DomainGeneralization.Table1.model import MyResnet, MyVGG, MyInceptionResnet, MyAlexnet
# from DomainGeneralization.Table1.PACSDataset import PACSDataset
import torchvision.transforms as transforms
# from train_teacher_map_pacs import DecoMapMNISTNN, EncoMapMNISTNN
# from da import *
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
# np.random.seed(cur_seed)
# random.seed(opt.seed)
# model_type = 'decoymnist'
# dset = 'dmnist'

wandb.init(project="change_fixed_new", entity="avani", config=opt,save_code=True)

# model_type = 'resnet50'
# dset = 'textures_train'
model_type = opt.model_type
dset = opt.dset


affect = 0
epochs = 50
affect_lis = [0]

lr = opt.lr
# lr = 1e-4
cdep_grad_method = opt.cdep_grad_method
regularizer_rate = opt.regularizer_rate
# num_imgs = opt.num_imgs
freeze_bef_layers = opt.freeze_bef_layers
pairs_vals =str(opt.pairs_vals)
wtcav = opt.wtcav
batch_size = opt.batch_size
knn_k = opt.knn_k
cur_proto_mean_wt = opt.cur_proto_mean_wt
use_proto = opt.use_proto
use_cdepcolor = opt.use_cdepcolor
use_precalc_proto = opt.use_precalc_proto
update_proto = opt.update_proto
train_from_scratch = opt.train_from_scratch
use_knn_proto= opt.use_knn_proto
class_wise_training = opt.class_wise_training
num_imgs = opt.num_imgs
bias = "1"
# if opt.use_wandb:
#     if use_proto:
#         wandb.init(project=dset+opt.loss_type+"_runs_new_"+opt.bottleneck_name+'proto_mean'+str(opt.do_proto_mean)+"wt"+str(cur_proto_mean_wt)+"pairs"+str(pair_num)+"affect"+str(affect), entity="avani")
#     else:
#         wandb.init(project=dset+opt.loss_type+"_runs_new_"+opt.bottleneck_name+'direct'+"pairs"+str(pair_num)+"affect"+str(affect), entity="avani")

#     wandb.init(config=opt)

print(opt.gpu_ids)
use_cuda = torch.cuda.is_available()

# from imdb_classi import *
# from imdb_classi_test import *
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
if model_type == 'resnet50':
    model = resnet50(pretrained=True)
    model_type = 'resnet50'
    model = model.cuda()
    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224,224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    bs = 16
    imagenet_zip_path = '/media/Data2/avani.gupta/Imagenet2012/Imagenet-sample/'
    train_dataset = ImageFolder(root=imagenet_zip_path+'train/', transform=val_transforms)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)

    val_dataset = ImageFolder(root=imagenet_zip_path+'val/', transform=val_transforms)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=True, num_workers=2)

    test_loader = None
    named_layers = dict(model.named_modules())
    lis = list(named_layers.keys())
    bottleneck_name = "layer4.2.conv3"
    sub_classes = os.listdir('/media/Data2/avani.gupta/Imagenet2012/Imagenet-sample/train/')

if model_type == 'faces':
    model = resnet18(pretrained=False)
    model.fc = nn.Linear(512,2)
    if not opt.train_from_scratch:
        model.load_state_dict(torch.load("/media/Data2/avani.gupta/bffhq/bffhq/bffhq_0.5_vanilla/result/best_model.th")['state_dict'])
    model.cuda()
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
    model.load_state_dict(torch.load('/home/avani.gupta/tcav_pt/cat_dog_model'+bias+'.pt'))
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

if model_type == 'clever_hans':
    import NeSyXIL.src.clevr_hans.cnn.data_xil as data
    # import NeSyXIL.src.clevr_hans.cnn.utils as utils
    import NeSyXIL.src.clevr_hans.cnn.model as model_
    data_dir = "/media/Data2/avani.gupta/CLEVR-Hans3/"

    
    if opt.dataset == "clevr-hans-state":
        dataset_train = data.CLEVR_HANS_EXPL(data_dir, "train", lexi=True, conf_vers='CLEVR-Hans3')
        dataset_val = data.CLEVR_HANS_EXPL(data_dir, "val", lexi=True, conf_vers='CLEVR-Hans3')
        dataset_test = data.CLEVR_HANS_EXPL(data_dir, "test", lexi=True, conf_vers='CLEVR-Hans3')
    else:
        print("Wrong dataset specifier")
        exit()

    opt.n_imgclasses = dataset_train.n_classes
    opt.classes = np.arange(opt.n_imgclasses)
    opt.category_ids = dataset_train.category_ids

    train_loader = data.get_loader(
        dataset_train,
        batch_size=batch_size,
        num_workers=opt.num_workers,
        shuffle=True,
    )
    train_proto_loader = data.get_loader(
        dataset_train,
        batch_size=512,
        num_workers=opt.num_workers,
        shuffle=True,
    )


    # test_loader = data.get_loader(
    #     dataset_test,
    #     batch_size=batch_size,
    #     num_workers=opt.num_workers,
    #     shuffle=False,
    # )
    # val_loader = data.get_loader(
    #     dataset_val,
    #     batch_size=batch_size,
    #     num_workers=opt.num_workers,
    #     shuffle=False,
    # )
    model = model_.ResNet34Small(num_classes=opt.n_imgclasses)
    if not train_from_scratch:
        st_dict = torch.load('/home/avani.gupta/tcav_pt/NeSyXIL/src/clevr_hans/cnn/runs/conf_3/resnet-clevr-hans-17-conf_3_seed0/model_epoch56_bestvalloss_0.0132.pth')['weights']
        model.load_state_dict(st_dict)
    model.cuda().eval()
    bottleneck_name = 'features.6.5.conv1'

if model_type == 'clever_hans7':
    import NeSyXIL.src.clevr_hans.cnn.data_xil as data
    # import NeSyXIL.src.clevr_hans.cnn.utils as utils
    import NeSyXIL.src.clevr_hans.cnn.model as model_
    data_dir = "/media/Data2/avani.gupta/CLEVR-Hans7/"
    
    if opt.dataset == "clevr-hans-state":
        dataset_train = data.CLEVR_HANS_EXPL(data_dir, "train", lexi=True, conf_vers='CLEVR-Hans7')
        dataset_val = data.CLEVR_HANS_EXPL(data_dir, "val", lexi=True, conf_vers='CLEVR-Hans7')
        dataset_test = data.CLEVR_HANS_EXPL(data_dir, "test", lexi=True, conf_vers='CLEVR-Hans7')
    else:
        print("Wrong dataset specifier")
        exit()

    opt.n_imgclasses = 7
    opt.classes = np.arange(opt.n_imgclasses)
    opt.category_ids = dataset_train.category_ids

    train_loader = data.get_loader(
        dataset_train,
        batch_size=batch_size,
        num_workers=opt.num_workers,
        shuffle=True,
    )
    train_proto_loader = data.get_loader(
        dataset_train,
        batch_size=1024,
        num_workers=opt.num_workers,
        shuffle=True,
    )
    # test_loader = data.get_loader(
    #     dataset_test,
    #     batch_size=batch_size,
    #     num_workers=opt.num_workers,
    #     shuffle=False,
    # )
    # val_loader = data.get_loader(
    #     dataset_val,
    #     batch_size=batch_size,
    #     num_workers=opt.num_workers,
    #     shuffle=False,
    # )
    model = model_.ResNet34Small(num_classes=opt.n_imgclasses)
    if not train_from_scratch:
        st_dict = torch.load('/home/avani.gupta/tcav_pt/runs/CLEVR-Hans7/2023-01-09_19:05:35-CLEVR-Hans7_seed10/model_epoch8_bestvalloss_0.2134.pth')['weights']
        model.load_state_dict(st_dict)
    model.cuda().eval()
    bottleneck_name = 'features.6.5.conv1'


if model_type == 'toy':
    biased_data, unbiased_data, labels = get_data()
    # labels = torch.from_numpy(labels).long().reshape(-1,1)
    unbiased_data = torch.from_numpy(np.float32(unbiased_data))
    labels = torch.from_numpy(labels).long()
    biased_data = torch.from_numpy(np.float32(biased_data))
    train_data = tutils.TensorDataset(biased_data,labels) 
    train_loader = tutils.DataLoader(train_data,batch_size=num_imgs,shuffle=True) 
    test_data = tutils.TensorDataset(unbiased_data,labels) 
    test_loader = tutils.DataLoader(test_data,batch_size=num_imgs)
    shape = (200,200)
    y_full = labels
    X_full = biased_data 
    model = Net()
    if not train_from_scratch:
        model.load_state_dict(torch.load('/media/Data2/avani.gupta/net.pt'))
    bottleneck_name = 'fc2'
    model.cuda()

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
        # if opt.load_best_finetuned:
        # model.load_state_dict(torch.load('4colormnistconv2loss_cos_msescratch_trainregsvmwcav0.5triplet_1cav_1new_0.5protolag_affectTruepairs3_knn_k3_cdep'))
        # else:
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
        # if opt.load_best_finetuned:
        #     model.load_state_dict(torch.load('4colormnistconv2loss_cos_msescratch_trainregsvmwcav0.5triplet_1cav_1new_0.5protolag_affectTruepairs3_knn_k3_cdep'))
        # else:
        model.load_state_dict(torch.load('/home/avani.gupta/tcav_pt/methodvanilla_tex.pt'))

s = ""
if train_from_scratch:
    s = "scratch_train"
else:
    s = "finetune"

root = "/media/Data2/avani.gupta"
full_root = root +''
device = 'cuda:'+str(0)


# bs = opt.batch_size
test_code = opt.test_code

finetune = opt.finetune
get_tcav_score = opt.get_tcav_score
num_random_exp = opt.num_random_exp
# model_type = model_type
use_triplet_loss = opt.use_triplet_loss

# bottleneck_name = 'conv2'

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

#step: get pairs: positive negative for concepts to be tested
# concepts = ['reflectance_large','shading_large','colors_small', 'Cube_happy_stanford-bunnylight0temp6500']
pairs_lis = []
affect_lis = []
# if opt.use_multi_pairs:
for pair_n in pairs_vals.split(","):
    # try:
    # if opt.use_num_imgs:
    with open('pairs_neww/pairs'+str(pair_n)+'_large_dino_mapped_to_stu_space_imgs'+str(num_imgs)+dset+"M_ep"+str(opt.mapping_mod_epoch)+'.pickle', 'rb') as handle:
            pairs_lis.append(pickle.load(handle))

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
loss_type = opt.loss_type
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

if opt.finetune:
    start_ep = 0
# if opt.use_multi_pairs:
#     affect = "and".join(affect_lis)
#     pair_num = "and".join(pair_num)
model_save_name = model_type+dset
k = ["pairs_vals","num_imgs","lr","regularizer_rate","wtcav","batch_size","cur_proto_mean_wt","use_proto","use_cdepcolor","use_precalc_proto","update_proto","train_from_scratch","use_knn_proto","class_wise_training"]
# for p in k:
#     model_save_name += str(p)+str(round(dic[p],2))
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

# k = ["pairs_vals","num_imgs","lr","regularizer_rate","wtcav","batch_size","cur_proto_mean_wt","use_proto","use_cdepcolor","use_precalc_proto","update_proto","train_from_scratch","use_knn_proto","class_wise_training"]
# for p in k:
#     model_save_name += str(p)+str(round(wandb.config[p],2))
print("round saving models ",model_save_name)

# if opt.do_proto_mean:
#     model_save_name = exp_name+s+"reg"+opt.reg_type+"wcav"+str(wtcav)+"triplet_"+str(opt.use_triplet_loss)+"cav_"+str(opt.use_cav_loss)
# else:
#     model_save_name = exp_name+s+"reg"+opt.reg_type+"wcav"+str(wtcav)+"triplet_"+str(opt.use_triplet_loss)+"cav_"+str(opt.use_cav_loss)
# if opt.smoothness_finetune:
#     model_save_name += "smoothness_wt"+str(opt.swtcav)
# model_save_name  += "affect"+affect
# model_save_name += "pairs"+pairs_vals
# if use_proto:
#     model_save_name += "proto"
#     if use_knn_proto:
#         model_save_name += "_knn_k"+str(knn_k)
# else:
#     model_save_name += "direct"
# if update_proto:
#     model_save_name += 'update_proto'
# else:
#     model_save_name += 'static_proto'
# if opt.do_proto_mean:
#     model_save_name += 'delta_proto'
# model_save_name += 'new_mapping'  
# if opt.load_best_finetuned:
#     model_save_name += 'best_finetuned_load'
# if not class_wise_training:
#      model_save_name += 'no_classwise'
# model_save_name += 'cor_nll_loss'
# if use_cdepcolor:
#     model_save_name += 'cdep'
#     model_save_name += 'reg_rate'+str(regularizer_rate)
# model_save_name += "num_imgs"+str(num_imgs)

# import time
#     timestr = time.strftime("%Y%m%d-%H%M%S")
# model_save_name += "time"+str(timestr)
    
res = {}
import torch.optim as optim
optimizer = optim.Adam(model.parameters(), lr=opt.lr)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
#     factor=0.1, patience=10, threshold=0.0001, threshold_mode='abs')

if model_type in ['toy','colormnist','decoymnist',"texturemnist"]:
    class_labels = y_full.unique().numpy()
    # for c in class_labels:
    #     idx, = torch.where(y_full==c)
    #     lb_wise_X[c] = X_full[idx] 

cos = nn.CosineSimilarity(dim=0,eps=1e-6)


proto_dic = {}
best_val_acc = -1
corr_test_acc = 0
for epoch in tqdm(range(epochs)):
    triplet_loss_lis = []
    cav_loss_lis = []
    if use_proto:
        if epoch==0 and use_precalc_proto:
            # if use_knn_proto:
            proto_dic = np.load('/media/Data2/avani.gupta/proto_'+model_type+"knn"+str(knn_k)+'.npy',allow_pickle=True)[()] #load proto dicts
            # else:
            #     proto_dic = np.load('/media/Data2/avani.gupta/proto_mean'+model_type+'.npy',allow_pickle=True)[()] #load proto dicts

        elif epoch%2==0 and update_proto:
            bss = 1000
            if 'clever_hans' in model_type:
                data= next(iter(train_proto_loader))
                X_full, _, y_full, _, _, _ = data
                del data
                bss = 50

            # if model_type in ['toy','colormnist','decoymnist']:
            for c in class_labels:
                acts = []
                def save_activation_hook(mod, inp, out):
                    global bn_activation
                    bn_activation = out
                idx, = torch.where(y_full==c)
                if len(idx)>0:
                    handle = named_layers[bottleneck_name].register_forward_hook(save_activation_hook)
                    for i in range(0,len(X_full[idx]),bss):
                        if model_type == 'toy':
                            _ = model(X_full[idx][i:min(i+bss,len(X_full[idx]))].cuda().reshape(-1,200*200))
                        else:
                            _ = model(X_full[idx][i:min(i+bss,len(X_full[idx]))].cuda())

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

                    # else:
                    #     if epoch==0 or not opt.do_proto_mean:
                    #         proto_dic[c] = np.expand_dims(np.mean(acts,axis=0))
                    #     else:
                    #         proto_dic[c] = (1-cur_proto_mean_wt)*proto_dic[c]+ cur_proto_mean_wt*np.expand_dims(np.mean(acts,axis=0)) #proto-type is avg of all of that class activatinos
                        

    #proto acts as psuedo GT
    #train only over last k layers
    # or train only bottleneck: todo
    enco_layer = bottleneck_name
    lis = list(named_layers.keys())
    if freeze_bef_layers and not train_from_scratch:
        idx_bt = lis.index(enco_layer) #freeze layers untill here
        for i in range(idx_bt-2):
            for p in named_layers[lis[i]].parameters():
                p.requires_grad = False
    
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
        
     
        if model_type == 'colormnist' or model_type =='decoymnist' or model_type=='texturemnist':
            gt_loss = torch.nn.functional.nll_loss(out, gt.to(device))
        elif model_type =='faces':
            # label = gt[:, 0]
            gt_loss = criterion(out, gt.to(device))
        elif model_type == 'cat_dog':
            # breakpoint()
            # gt = gt.view(len(gt), 1)
            gt_loss= criterion_bce(out, gt.view(len(gt), 1).to(device))
        else:
            gt_loss = criterion(out,gt.to(device))


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
            elif model_type =='decoymnist':
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

                        #print(torch.cuda.max_memory_allocated(0)/np.power(10,9))
                        (regularizer_rate*add_loss +gt_loss).backward(retain_graph=True)
                    elif cdep_grad_method ==1:
                        add_loss +=score_funcs.gradient_sum(imgs, gt.to(device), torch.FloatTensor(blob).to(device),  model, torch.nn.functional.nll_loss)
                        (regularizer_rate*add_loss).backward(retain_graph=True)

                        #print(torch.cuda.max_memory_allocated(0)/np.power(10,9))
                        # optimizer.step()

                    elif cdep_grad_method ==2:
                        for j in range(len(imgs)):
                            add_loss +=(score_funcs.eg_scores_2d(model, imgs, j, gt.to(device), num_samples) * torch.FloatTensor(blob).to(device)).sum()
                        (regularizer_rate*add_loss).backward(retain_graph=True)

                        #print(torch.cuda.max_memory_allocated(0)/np.power(10,9))
                        # optimizer.step()
                        # loss = torch.nn.functional.nll_loss(out, target)
                        # loss.backward()


        if opt.use_gt_loss:
            gt_loss.backward(retain_graph=True)
            # optimizer.step()
        
        faces_class_wise = {0:"young_women",1:"old_men"}


        #calc grads for inputs
        if class_wise_training:
            for c in class_labels:
                idx, = torch.where(gt==c)
                imgs_class_wise = imgs[idx]
                if len(imgs_class_wise)==0:
                    continue
                for pn,pairs_all in enumerate(pairs_lis):
                    for main_con in pairs_all:
                        pairs = pairs_all[main_con]
                        # print(pairs, affect_lis[pn])

                        for qs,pair in enumerate(pairs):
                            import glob
                            affect = affect
                            concept = pair[0]
                            if 'women'==concept or concept=='men':
                                affect = 1 #men vs women concept shall affect
                            do = False
                            if model_type=='clever_hans' and ((c==0 and 'gray' in concept) or (c==1 and 'metal' in concept) or (c==2 and 'cube' in concept)):
                                do = True
                            elif model_type=='colormnist' and concept == str(c):
                                do = True
                            elif model_type == 'faces' and concept ==faces_class_wise[c]:
                                do = True

                            if do:
                                if opt.teach_mapped_to_stu:
                                    if model_type =='clever_hans' or model_type =='clever_hans7':
                                        fc_save_path = '/media/Data2/avani.gupta/fc_new/dino_mapped_to_stu_space_'+dset+'/'
                                        files =  glob.glob(fc_save_path+'_cons_'.join(pair)+'*.pt')
                                        save_path = files[0]
                                        dic = torch.load(save_path)
                                    else:
                                        fc_save_path = '/media/Data2/avani.gupta/fc_new/dino_mapped_to_stu_space_num_imgs'+str(num_imgs)+dset+'/'
                                        files =  glob.glob(fc_save_path+'_cons_'.join(pair)+'*.pt')
                                        save_path = files[0]
                                        dic = torch.load(save_path)
                                
                                if not isinstance(dic, dict):
                                    with torch.no_grad():
                                        cavs = dic
                                        del dic
                                else:
                                    with torch.no_grad():
                                        cavs = dic['weight'].cpu().numpy()
                                        del dic

                                direc_mean = torch.tensor(cavs[0])
                                direc_mean.requires_grad = True

                                named_layers = dict(model.named_modules())
                                bn_activation= None
                                grad = None
                                tcav_score = {}
                                results = []
                                ep_loss = 0
                                dot_lis = []
                                cos_lis = []
                                mse_dvec_lis = []

                                for b in range(imgs_class_wise.shape[0]):
                                    e = imgs_class_wise[b]
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
                                                loss = mse_loss(act, torch.from_numpy(proto_dic[c]['cluster_'+str(f)]).cuda().unsqueeze(0))
                                            else:
                                                loss += mse_loss(act, torch.from_numpy(proto_dic[c]['cluster_'+str(f)]).cuda().unsqueeze(0))
                                        # else:  
                                        #     loss = mse_loss(act, torch.from_numpy(proto_dic[c]).cuda()) #loss btw act and proto
                                    
                                        grad_ = torch.autograd.grad(loss, act, retain_graph=True,create_graph=True)
                                    else:
                                        grad_ = torch.autograd.grad(out[0][gt[b]], act, retain_graph=True,create_graph=True)

                                    dot = torch.dot(grad_[0].to(device).float().squeeze(0).flatten(),direc_mean.to(device).float())/torch.linalg.norm(direc_mean)
                                    dot_lis.append(dot)
                                    
                                    unit_grad = grad_[0].to(device).float().squeeze(0).flatten()
                                    unit_direc = direc_mean.to(device).float()/torch.linalg.norm(direc_mean)
                                    unit_grad = unit_grad/torch.linalg.norm(unit_grad)
                                    mse_dvec_lis.append(mse_loss(unit_grad, unit_direc))

                                    if 'cos' in loss_type:
                                        cos_ = cos(grad_[0].to(device).float().squeeze(0).flatten(),direc_mean.to(device).float())
                                        cos_lis.append(cos_)

                                    handle.remove()
                        
                                dot_lis = torch.stack(dot_lis)
                                mse_dvec_lis = torch.stack(mse_dvec_lis)
                                if 'cos' in loss_type:
                                    cos_lis = torch.stack(cos_lis)

                                #or cs(grad and direc_mean) max?
                                score = len(torch.where(dot_lis<0)[0])/len(dot_lis)
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
                                            # print(concept,tcav_loss)
                                            loss_ = wtcav*tcav_loss
                                loss_ = wtcav*loss_
                                cav_loss_lis.append(loss_.item())
                                if opt.use_cav_loss:
                                    if qs==len(pairs):
                                        loss_.backward()
                                        print(grad)
                                    else:
                                        loss_.backward(retain_graph=True)


        else:
            for pn,pairs_all in enumerate(pairs_lis):
                for main_con in pairs_all:
                    pairs = pairs_all[main_con]
                    # print(pairs, affect_lis[pn])
                    for qs,pair in enumerate(pairs):
                        import glob
                        affect = affect
                        
                        
                        concept = pair[0]
                        if concept=='women' or concept=='men':
                            affect = 1 #men vs women concept shall affect

                        if opt.teach_mapped_to_stu:
                           
                            fc_save_path = '/media/Data2/avani.gupta/fc_new/dino_mapped_to_stu_space_num_imgs'+str(num_imgs)+dset+"M_ep"+str(opt.mapping_mod_epoch)+'/'
                            # fc_save_path = '/media/Data2/avani.gupta/fc_new/dino_mapped_to_stu_space_'+dset+'/'
                        
                        files =  glob.glob(fc_save_path+'_cons_'.join(pair)+'*.pt')

                        save_path = files[0]

                        dic = torch.load(save_path)
                        if not isinstance(dic, dict):
                            with torch.no_grad():
                                cavs = dic
                                del dic
                        else:
                            with torch.no_grad():
                                cavs = dic['weight'].cpu().numpy()
                                del dic

                        direc_mean = torch.tensor(cavs[0])
                        direc_mean.requires_grad = True

                        named_layers = dict(model.named_modules())
                        bn_activation= None
                        grad = None
                        tcav_score = {}
                        results = []
                        ep_loss = 0
                        dot_lis = []
                        cos_lis = []
                        mse_dvec_lis = []

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
                                grad_ = torch.autograd.grad(out[0][gt[b]], act, retain_graph=True,create_graph=True)
                            # loss = criterion(out,gt[b].unsqueeze(0).to(device))
                            # if loss_type=='mse':
                            dot = torch.dot(grad_[0].to(device).float().squeeze(0).flatten(),direc_mean.to(device).float())/torch.linalg.norm(direc_mean)
                            dot_lis.append(dot)
                            
                            unit_grad = grad_[0].to(device).float().squeeze(0).flatten()
                            unit_direc = direc_mean.to(device).float()/torch.linalg.norm(direc_mean)
                            unit_grad = unit_grad/torch.linalg.norm(unit_grad)
                            mse_dvec_lis.append(mse_loss(unit_grad, unit_direc))

                            if 'cos' in loss_type:
                                cos_ = cos(grad_[0].to(device).float().squeeze(0).flatten(),direc_mean.to(device).float())
                                cos_lis.append(cos_)

                            handle.remove()
                
                        dot_lis = torch.stack(dot_lis)
                        mse_dvec_lis = torch.stack(mse_dvec_lis)
                        if 'cos' in loss_type:
                            cos_lis = torch.stack(cos_lis)

                        #or cs(grad and direc_mean) max?
                        score = len(torch.where(dot_lis<0)[0])/len(dot_lis)
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
                                    # print(concept,tcav_loss)
                                    loss_ = wtcav*tcav_loss
                        loss_ = wtcav*loss_
                        cav_loss_lis.append(loss_.item())
                        if opt.use_cav_loss:
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
                # if model_type =='cat_dog': #has logits
                #     val_loss += criterion_bce(output, target.view(len(target), 1).to(device)).item()
                # else: #other dsets have softmax
                #     test_loss += torch.nn.functional.nll_loss(output, target, reduction='sum').item() # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        # test_loss /= len(test_loader.dataset)
        test_acc = 100. * correct / len(test_loader.dataset)
        # if opt.use_wandb:
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


        # if opt.use_wandb:
        wandb.log({"gt_loss": gt_loss, "cav_loss":sum(cav_loss_lis) ,"val_gt_loss":val_loss, "val_acc":val_acc, "test_gt_loss":test_loss,"test_acc":test_acc,"best_val_acc":best_val_acc, "corr_test_acc":corr_test_acc})#,"small_num_op_loss":small_num_op_loss_mean})

        # wandb.log({"gt_loss": gt_loss, "cav_loss":sum(cav_loss_lis) ,"val_gt_loss":val_loss, "val_acc":val_acc, "test_gt_loss":test_loss,"test_acc":test_acc,"best_val_acc":best_val_acc, "corr_test_acc":corr_test_acc})#,"small_num_op_loss":small_num_op_loss_mean})
        # wandb.log({"gt_loss": gt_loss, "cav_loss":sum(cav_loss_lis)})#,"small_num_op_loss":small_num_op_loss_mean})
        cav_loss_lis = []
        # torch.save(model.state_dict(), os.path.join(opt.checkpoints_dir+'sweeps',"acc"+str(test_acc)+"best"+str(epoch)+model_save_name+"final_run"))

    # torch.save(model.state_dict(), os.path.join(opt.checkpoints_dir,str(epoch)+model_save_name+"final_run"))

print("we are done!!!!!")
# torch.save(model.state_dict(), str(epoch)+model_save_name)
# print("res on arap",get_mse_arap(model,load_model=False))