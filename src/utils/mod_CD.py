from utils_train import * 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import glob
import os
import copy
import numpy as np
from collections import OrderedDict
from sklearn.cluster import KMeans
from tqdm import tqdm
from statistics import mean
import torch.utils.data as tutils
from torchvision.models import resnet18
import torchvision.transforms as transforms



class CD():
    '''
    Concept Distillation (CD)
    '''
    def __init__(self, input_shape, num_classes, num_domains=0, hparams=None,network=None):
#         super().__init__(input_shape, num_classes, num_domains=0, hparams=None,network=None)
        self.proto_update_last = 0
        self.network =  network #nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["lr"],
        )
        self.hparams = hparams
        self.mse_loss = nn.MSELoss()
        self.cos = nn.CosineSimilarity(dim=0,eps=1e-6)
        self.input_shape = input_shape
        self.model_type = hparams['model_type']
        self.dset = hparams['dset']
        self.num_imgs = hparams['num_imgs']
        self.pairs_vals = str(hparams['pairs_vals'])
        self.pairs_lis = [get_pairs(self.model_type,self.pairs_vals)]
        self.bottleneck_name = hparams["bottleneck_name"]
        self.affect_lis = hparams['affect_lis'] #pair-wise affect list
        self.proto_dic = {}
        self.class_labels = np.arange(0, num_classes)
        self.named_layers = dict(self.network.named_modules()) 
        self.map_type = 'S_to_T'
        self.num_imgs = num_imgs



    def save_student_outs(self,pairs_vals):
        """
        Step 1 for training mapping module
        """
        img_data = ImgsDataset(self.model_type,self.input_shape,concept_set_type=str(pairs_vals),num_imgs = self.num_imgs)
        dataloader = DataLoader(img_data, batch_size=1,
                                shuffle=True, num_workers=0)

        save_path = "/media/Data2/avani.gupta/acts_"+self.model_type+self.bottleneck_name+'/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for i, (img_data, con) in enumerate(dataloader):
            img_data = torch.permute(img_data.squeeze(0), (0, 3, 2, 1))
            acts = []
            for im in img_data:
                bn_activation = None
                def save_activation_hook(mod, inp, out):
                    nonlocal bn_activation
                    bn_activation = out
                handle = self.named_layers[self.bottleneck_name].register_forward_hook(save_activation_hook)
                out = self.network(im.unsqueeze(0).cuda().float())
                act = bn_activation.detach()
                acts.append(act.unsqueeze(0))
                handle.remove()
            acts = torch.concat(acts,axis=0)
            torch.save(acts,save_path+con[0]+'all.pt')
            print("acts",acts.shape)
            print(f"saved {acts.shape} to {save_path+con[0]+'all.pt'}")


    def map_activation_spaces(self):
        """
        Step 2: training mapping module to map activation spaces of teacher and student
        """
        concepts = set()
        for pairs in self.pairs_lis:
            for p in pairs:
                concepts.add(p)
                for l1,l2 in pairs[p]:
                    concepts.add(l1)
                    concepts.add(l2)
        concepts = list(concepts)
        downconv_m = EncoMapResnet50NN()
        upconv_m = DecoMapResnet50NN()
        mse_loss = nn.MSELoss()
        upconv_m.train()
        downconv_m.train()
        upconv_m.cuda()
        downconv_m.cuda()
        model_params = list(upconv_m.parameters())+list(downconv_m.parameters())
        optimizer = torch.optim.Adam([p for p in model_params if p.requires_grad], lr=1e-4)
        loss_lis = []
        first = 1
        for ep in tqdm(range(2)):
            for con in tqdm(concepts):
                # if not os.path.exists("/media/Data2/avani.gupta/acts_"+self.model_type+self.bottleneck_name+'/'+con+'all.pt'):
                #     print("calling save student outs")
                #     self.save_student_outs()
                
                img_fea = torch.load('/media/Data2/avani.gupta/dino_fea_'+str(self.input_shape[0])+'by'+str(self.input_shape[1])+'/'+con+'all.pt').cuda()
                stu_pred = torch.load("/media/Data2/avani.gupta/acts_"+self.model_type+self.bottleneck_name+'/'+con+'all.pt').cuda()
                fea_s = img_fea.shape
                for i,img_ in enumerate(img_fea):
                    x = downconv_m(img_)
                    x_= upconv_m(x)
                    stu_pred_ = stu_pred[i].squeeze(0)
                    assert(img_.shape==x_.shape)
                    assert(x.shape==stu_pred_.shape)
                    loss = mse_loss(img_, x_) + mse_loss(x, stu_pred_)
                    if i!=len(img_fea)-1:
                        loss.backward(retain_graph=True)
                    else:
                        loss.backward()
                    assert(torch.isnan(torch.tensor(loss.item()))==False)
                    loss_lis.append(loss.item())
                optimizer.step()
                optimizer.zero_grad()
            print("ep",ep," loss",mean(loss_lis))

        path = '/media/Data2/avani.gupta/teacher_2/'+self.dset+'pairs_vals'+str(self.pairs_vals)
        if not os.path.exists(path):
            os.makedirs(path)

        torch.save(upconv_m.state_dict(), path+'upconv.pt')
        torch.save(downconv_m.state_dict(), path+'downconv.pt')

    def train_cavs(self):
        """
        Step 3 training CAV in teacher/student space; all other components frozen including mapping module
        """
        if self.map_type=='T_to_S':
            if self.dset == 'pacs':
                map_nn = DecoMapResnet50NN()
                path ='/media/Data2/avani.gupta/teacher_2/'+self.dset+'pairs_vals'+str(self.pairs_vals)
                map_nn.load_state_dict(torch.load(path+'downconv.pt'))
                map_nn.eval()

        new_pairs = {}
        for pairs_main in self.pairs_lis:
            for main_con in pairs_main:
                pairs = pairs_main[main_con]
                for pair in pairs:
                    con, neg_con = pair
                    
                    acts = {}
                    acts[con] = torch.load('/media/Data2/avani.gupta/dino_fea_'+str(self.input_shape[1])+'by'+str(self.input_shape[2])+'/'+con+'all.pt')
                    acts[neg_con] = torch.load('/media/Data2/avani.gupta/dino_fea_224by224/'+neg_con+'all.pt')
                    if self.map_type=='T_to_S':
                        with torch.no_grad():
                            acts[con] = map_nn(acts[con])
                            acts[neg_con] = map_nn(acts[neg_con])

                    cav, fc_state_dict = get_or_train_cav_nn([con, neg_con], 'dino', acts)
                    
                    fc_save_path = ''
                    if self.map_type=='T_to_S':
                        fc_save_path = '/media/Data2/avani.gupta/fc_new/dino_mapped_to_stu_space_num_imgs'+str(self.num_imgs)+self.dset+'/'
                    else:
                        fc_save_path = '/media/Data2/avani.gupta/fc_new/direct_dino_pca_space_num_imgs'+str(self.num_imgs)+self.dset+'/'
                    
                    if not os.path.exists(fc_save_path):
                        os.makedirs(fc_save_path)

                    if cav.accuracies['overall'] > 0.6:
                        torch.save(fc_state_dict, fc_save_path+''+'_cons_'.join(pair)+str(cav.accuracies["overall"])+'fc.pt')  
                        if con in new_pairs:
                            new_pairs[con].append((con,neg_con))
                        else:
                            new_pairs[con] = [pair]
                    
                print(new_pairs)
                if self.map_type=='T_to_S':
                    with open('pairs_neww/pairs'+str(self.pairs_vals)+'_large_dino_mapped_to_stu_space_imgs'+str(self.num_imgs)+self.dset+'.pickle', 'wb') as handle:
                        pickle.dump(new_pairs, handle, protocol=pickle.HIGHEST_PROTOCOL) 
                else:
                    with open('pairs_neww/pairs'+str(self.pairs_vals)+'_large_direct_dino_pca_space_num_imgs'+str(self.num_imgs)+self.dset+'.pickle', 'wb') as handle:
                        pickle.dump(new_pairs, handle, protocol=pickle.HIGHEST_PROTOCOL)                  


    


   
    def update(self, minibatches,mega_minibatches, unlabeled=None):
        if self.map_type=='S_to_T':
            path = '/media/Data2/avani.gupta/teacher_2/'+self.dset+'pairs_vals'+str(self.pairs_vals)
            self.upconv_m = DecoMapResnet50NN()
            self.upconv_m.load_state_dict(torch.load(path+'upconv.pt'))
            
        if self.proto_update_last==0 or np.random.random()>self.hparams['proto_update_prob']:
            self.calc_proto(mega_minibatches)
            self.proto_update_last = 1
        running_loss = 0
        for x, y in minibatches:
            self.optimizer.zero_grad()
            gt_loss = F.cross_entropy(self.predict(x), y)
            gt_loss.backward(retain_graph=True)
            running_loss += gt_loss.item()
            for pn,pairs_all in enumerate(self.pairs_lis):
                for main_con in pairs_all:
                    pairs = pairs_all[main_con]
                    for qs,pair in enumerate(pairs):
                        affect = self.affect_lis[qs]
                        c_loss = self.get_concept_loss(x,y,pair,affect)
                        if pn==len(self.pairs_lis)-1:
                            c_loss.backward()
                            running_loss+= c_loss.item()
                        else:
                            c_loss.backward(retain_graph=True)
                            running_loss+= c_loss.item()
            self.optimizer.step()
            return {'loss': running_loss}

    def calc_proto(self,batches):
        X_full = torch.cat([x for x, y in batches])
        y_full = torch.cat([y for x, y in batches])
        bn_activation = None
        for c in self.class_labels:
            acts = []
            def save_activation_hook(mod, inp, out):
                nonlocal bn_activation
                bn_activation = out
            idx, = torch.where(y_full==c)
            bss = 32
            if len(idx)>self.hparams["knn_k"]:
                handle = self.named_layers[self.bottleneck_name].register_forward_hook(save_activation_hook)
                for i in range(0,len(X_full[idx]),bss):
                    _ = self.predict(X_full[idx][i:min(i+bss,len(X_full[idx]))].cuda(0))
                    acts.append(bn_activation.detach().cpu())
                acts = torch.concat(acts,axis=0)
                ss = acts.shape

                #giving half weightage to old proto mean and half to new proto
                # if use_knn_proto:
                acts = acts.cpu().numpy().reshape((-1,ss[1]*ss[2]*ss[3]))
                kmeans = KMeans(n_clusters=self.hparams["knn_k"], random_state=0).fit(acts)
                centers = kmeans.cluster_centers_
                for f in range(self.hparams["knn_k"]):
                    if not self.hparams["do_proto_mean"]:
                        if c in self.proto_dic:
                            self.proto_dic[c]['cluster_'+str(f)] = centers[f].reshape(ss[1],ss[2],ss[3])
                        else:
                            self.proto_dic[c] = {}
                            self.proto_dic[c]['cluster_'+str(f)] = centers[f].reshape(ss[1],ss[2],ss[3])
                    else:
                        if c in self.proto_dic and 'cluster_'+str(f) in self.proto_dic[c]:
                            self.proto_dic[c]['cluster_'+str(f)] = (1-self.hparams["cur_proto_mean_wt"])*self.proto_dic[c]['cluster_'+str(f)]+self.hparams["cur_proto_mean_wt"]*centers[f].reshape((ss[1],ss[2],ss[3]))
                        elif c in self.proto_dic:
                            self.proto_dic[c]['cluster_'+str(f)] = centers[f].reshape(ss[1],ss[2],ss[3])
                        else:
                            self.proto_dic[c] = {}
                            self.proto_dic[c]['cluster_'+str(f)] = centers[f].reshape(ss[1],ss[2],ss[3])

    def get_concept_loss(self,imgs,gt,pair,affect): 
        named_layers = dict(self.network.named_modules())            
        fc_save_path = '/media/Data2/avani.gupta/fc_new/dino_mapped_to_stu_space_num_imgs'+str(self.num_imgs)+self.dset+'/'
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

        bn_activation= None
        dot_lis = []
        cos_lis = []
        mse_dvec_lis = []
        device = 'cuda:0'
        bn_activation = None
        for b in range(imgs.shape[0]):
            e = imgs[b]
            def save_activation_hook(mod, inp, out):
                nonlocal bn_activation
                bn_activation = out
            handle = named_layers[self.bottleneck_name].register_forward_hook(save_activation_hook)
            out = self.network(e.unsqueeze(0).cuda().float())
            act = bn_activation
            grad_ = None
            #mapping stuff
            #teacher mapped to student third last layer
            if self.map_type=='T_to_S':
                pass
            else:
                act = self.upconv_m(act)

            if self.hparams["use_proto"]:
                for f in range(self.hparams["knn_k"]):
                    if f==0:
                        loss_a = self.mse_loss(act, torch.from_numpy(self.proto_dic[gt[b].item()]['cluster_'+str(f)]).cuda().unsqueeze(0))
                    else:
                        loss_a += self.mse_loss(act, torch.from_numpy(self.proto_dic[gt[b].item()]['cluster_'+str(f)]).cuda().unsqueeze(0))
        
                grad_ = torch.autograd.grad(loss_a, act, retain_graph=True,create_graph=True)
                
            else:
                grad_ = torch.autograd.grad(out[0][gt[b]], act, retain_graph=True,create_graph=True)
           
            dot = torch.dot(grad_[0].to(device).float().squeeze(0).flatten(),direc_mean.to(device).float())/torch.linalg.norm(direc_mean)
            dot_lis.append(dot)
            
            unit_grad = grad_[0].to(device).float().squeeze(0).flatten()
            unit_direc = direc_mean.to(device).float()/torch.linalg.norm(direc_mean)
            unit_grad = unit_grad/torch.linalg.norm(unit_grad)
            mse_dvec_lis.append(self.mse_loss(unit_grad, unit_direc))

            cos_ = self.cos(grad_[0].to(device).float().squeeze(0).flatten(),direc_mean.to(device).float())
            cos_lis.append(cos_)

            handle.remove()

        dot_lis = torch.stack(dot_lis)
        mse_dvec_lis = torch.stack(mse_dvec_lis)
        cos_lis = torch.stack(cos_lis)
        
        if affect==False:
            loss_ = torch.sum(torch.abs(cos_lis)) #L1
        if affect==True:
            loss_= self.mse_loss(cos_lis, torch.ones(len(cos_lis)).to(device))
        return loss_


    def predict(self, x):
        return self.network(x)

from argparse import ArgumentParser
import sys
parser = ArgumentParser()
parser.add_argument("--knn_k", default=3, type=int, help="knn value")
parser.add_argument("--cur_proto_mean_wt", default=0.3, type=float)
parser.add_argument("--proto_update_prob", default=0.2, type=float)
parser.add_argument("--use_proto", default=False, type=bool)
parser.add_argument("--do_proto_mean", default=True, type=bool)
parser.add_argument("--model_type", default='colormnist')
parser.add_argument("--dset", default='mnist')
parser.add_argument("--num_imgs", default=150, type=int)
parser.add_argument("--lr", default=0.1, type=float)
parser.add_argument("--pairs_vals", default=8, type=int)
parser.add_argument("--affect_lis", default=0, type=int)
parser.add_argument("--train_from_scratch", default=1, type=int)
parser.add_argument("--bottleneck_name", default='conv2', type=str)

# sys.argv = ['']
# parser.set_defaults(verbose=False)
hparams = parser.parse_args()
# from da import *
from imdb_classi_test import *
from networks_classi import *

model_type = hparams.model_type
num_imgs = hparams.num_imgs
pairs_vals = hparams.pairs_vals
batch_size = 64
if model_type == 'faces':
    model = resnet18(pretrained=False)
    model.fc = nn.Linear(512,2)
    if not hparams.train_from_scratch:
        model.load_state_dict(torch.load("/media/Data2/avani.gupta/bffhq/bffhq/bffhq_0.5_vanilla/result/best_model.th")['state_dict'])
    model.cuda()
    model.eval()
    bs = 64
    train_loader = DataLoader(bFFHQDataset("train"),bs, shuffle=True, num_workers=0)
    val_loader = DataLoader(bFFHQDataset("valid"),bs, shuffle=True, num_workers=0)
    test_loader = DataLoader(bFFHQDataset("test"),bs, shuffle=True, num_workers=0)
    named_layers = dict(model.named_modules())
    lis = list(named_layers.keys())
    bottleneck_name = 'layer4.1.conv1'
    shape = (224,224)
    num_classes = 2
    full_X_proto = False



if model_type =='cat_dog':
    full_X_proto = False

    model = models.resnet18(pretrained=True)
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, 1)
    bias = 'TB'+str(bias)
    model.load_state_dict(torch.load('/home/avani.gupta/tcav_pt/cat_dog_model'+bias+'.pt'))
    model = model.cuda()
    model.eval()
    shape = (224,224)

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
    num_clsses = 2


if model_type == 'decoymnist':
    dpath = 'dep/data/DecoyMNIST/'
    X_full = torch.Tensor(np.load(os.path.join(dpath, "train_x_decoy.npy")))
    y_full = torch.Tensor(np.load(os.path.join(dpath, "train_y.npy"))).type(torch.int64)
    complete_dataset = tutils.TensorDataset(X_full, y_full) # create your datset

    num_train = int(len(complete_dataset)*.9)
    num_test = len(complete_dataset)  - num_train 
    torch.manual_seed(0)
    train_dataset, val_dataset,= torch.utils.data.random_split(complete_dataset, [num_train, num_test])
    train_loader = tutils.DataLoader(train_dataset, batch_size=num_imgs, shuffle=True, **kwargs) # create your dataloader
    val_loader = tutils.DataLoader(val_dataset, batch_size=num_imgs, shuffle=True, **kwargs) # create your dataloader

    test_x_tensor = torch.Tensor(np.load(os.path.join(dpath, "test_x_decoy.npy")))
    test_y_tensor = torch.Tensor(np.load(os.path.join(dpath, "test_y.npy"))).type(torch.int64)
    test_dataset = tutils.TensorDataset(test_x_tensor,test_y_tensor) # create your datset
    test_loader = tutils.DataLoader(test_dataset, batch_size=num_imgs, shuffle=True, **kwargs) # create your dataloader
    
    model = MNISTDecoyNet()
    model.cuda()
    shape = (28,28)
    if not hparams.train_from_scratch:
        model.load_state_dict(torch.load('mnist/DecoyMNIST/orig_model_decoyMNIST_.pt'))
    bottleneck_name = 'conv2'
    num_classes = 10
    full_X_proto = True


if model_type =='colormnist':
    x_numpy_train = np.load(os.path.join("dep/data/ColorMNIST", "train_x.npy"))
    prob = (x_numpy_train.sum(axis = 1) > 0.0).mean(axis = 0).reshape(-1)
    prob /=prob.sum()
    mean = x_numpy_train.mean(axis = (0,2,3))
    std = x_numpy_train.std(axis = (0,2,3))
    bottleneck_name = hparams.bottleneck_name
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

    if not hparams.train_from_scratch:
        # if hparams.load_best_finetuned:
        #     model.load_state_dict(torch.load('4colormnistconv2loss_cos_msescratch_trainregsvmwcav0.5triplet_1cav_1new_0.5protolag_affectTruepairs3_knn_k3_cdep'))
        # else:
        model.load_state_dict(torch.load('mnist/ColorMNIST/orig_model_colorMNIST.pt'))
    num_classes = 10
    full_X_proto = True


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
    num_classes = 10
    full_X_proto = True
    

    if not hparams.train_from_scratch:
        model.load_state_dict(torch.load('/home/avani.gupta/tcav_pt/methodvanilla_tex.pt'))

hparams = vars(hparams)
# cd = CD(shape, num_classes,hparams=hparams,network=model)#full_X_proto=full_X_proto)
cd = CD(shape, num_classes,hparams=hparams,network=model)#full_X_proto=full_X_proto)
cd.save_student_outs(pairs_vals=pairs_vals)
cd.map_activation_spaces()
cd.train_cavs()