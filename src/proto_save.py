from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import sys
sys.path.append("../")
import torch
from tqdm import tqdm
import os
import numpy as np
from tqdm import tqdm
torch.cuda.empty_cache()
from sklearn.cluster import KMeans

from utils.utils_train import *
from utils.networks_classi import *
from utils.imdb_classi_test import *
from utils.utils_tcav2 import *
from utils.tcav_options import *


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

opt = TCAVOptions().parse() 
model_type = opt.model_type
knn_k = opt.knn_k
dset = opt.knn_k
affect = opt.pair_affect
num_imgs = opt.num_imgs
bias = opt.bias
kwargs = {'num_workers': 1, 'pin_memory': True} if 1 else {}
sample_type = '0'

print(opt.gpu_ids)
use_cuda = torch.cuda.is_available()


exp_name = model_type+opt.bottleneck_name+"loss_"+opt.loss_type

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
kwargs = {'num_workers': 1, 'pin_memory': True}
model, bottleneck_name, train_loader, val_loader, test_loader, X_full, y_full, val_x, val_y, shape = get_model_and_data(model_type, opt, kwargs)
named_layers = dict(model.named_modules())
lis = list(named_layers.keys())

proto_dic = {}
bss = 100
if opt.model_type =='clever_hans':
    class_labels = [0,1] #for lb 0 gray concept should not affect, for lb 1 metal should not
elif opt.model_type =='clever_hans7':
    class_labels = [0,1,2]
if model_type in ['toy','colormnist','decoymnist','texturemnist','cifar10']:
    class_labels = y_full.unique().numpy()


for c in tqdm(class_labels):
    acts = []
    def save_activation_hook(mod, inp, out):
        global bn_activation
        bn_activation = out
    idx, = torch.where(y_full==c)
    handle = named_layers[bottleneck_name].register_forward_hook(save_activation_hook)
    for i in range(0,len(X_full[idx]),bss):
        with torch.no_grad():
            if model_type == 'toy':
                _ = model(X_full[idx][i:min(i+bss,len(X_full[idx]))].cuda().reshape(-1,200*200))
            else:
                _ = model(X_full[idx][i:min(i+bss,len(X_full[idx]))].cuda())
        acts.append(bn_activation)
    acts = torch.concat(acts,axis=0)
    ss = acts.shape

    if opt.use_knn_proto and opt.knn_k!=1:
        acts = acts.detach().cpu().numpy().reshape((-1,ss[1]*ss[2]*ss[3]))
        kmeans = KMeans(n_clusters=opt.knn_k, random_state=0).fit(acts)
        centers = kmeans.cluster_centers_
        for f in range(opt.knn_k):
            # if not opt.do_proto_mean:
            if c in proto_dic:
                proto_dic[c]['cluster_'+str(f)] = centers[f].reshape(ss[1],ss[2],ss[3])
            else:
                proto_dic[c] = {}
                proto_dic[c]['cluster_'+str(f)] = centers[f].reshape(ss[1],ss[2],ss[3])
           
    else:
        proto_dic[c] = torch.mean(acts,axis=0).unsqueeze(0).detach().cpu().numpy()

if opt.use_knn_proto:
    np.save('/media/Data2/avani.gupta/proto_'+model_type+"knn"+str(opt.knn_k)+'.npy',proto_dic)
    print('saved to /media/Data2/avani.gupta/proto_'+model_type+"knn"+str(opt.knn_k)+'.npy')
else:
    np.save('/media/Data2/avani.gupta/proto_mean'+model_type+'.npy',proto_dic)
    print(f"saved to /media/Data2/avani.gupta/proto_mean+{model_type}+'.npy")


