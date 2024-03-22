from utils.utils_train import * 
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
from torchvision.models import resnet50
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import dep.score_funcs as score_funcs
DATA_PATH='/ssd_scratch/cvit/avani.gupta/'

class CD():
    '''
    Concept Distillation (CD)
    '''
    def __init__(self, input_shape, num_domains=0, hparams=None,network=None, class_labels = [], transform=None):
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
        self.named_layers = dict(self.network.named_modules()) 
        self.map_type = 'T_to_S'
        self.num_imgs = num_imgs
        self.class_labels = class_labels
        self.transform = transform
        self.write = True
        if self.hparams["use_proto"]:
            self.proto_dic = np.load(DATA_PATH+'proto_'+hparams["model_type"]+"knn"+str(hparams["knn_k"])+'.npy',allow_pickle=True)[()] #load proto dicts
        
        # if self.map_type=='S_to_T':
        path = DATA_PATH+'teacher_2/'+self.dset+'pairs_vals'+str(self.pairs_vals)
        self.upconv_m, self.downconv_m, _,_ = get_mapping_module(model_type)
        if os.path.exists(path+'downconv.pt'):
            self.downconv_m.load_state_dict(torch.load(path+'downconv.pt'))
            self.downconv_m = self.downconv_m.cuda()

        if self.map_type=='T_to_S':
            self.pairs_path = 'pairs_neww/pairs'+str(self.pairs_vals)+'_large_dino_mapped_to_stu_space_imgs'+str(self.num_imgs)+self.dset+'.pickle'
        else:
            self.pairs_path = 'pairs_neww/pairs'+str(self.pairs_vals)+'_large_direct_dino_pca_space_num_imgs'+str(self.num_imgs)+self.dset+'.pickle'
        
        if os.path.exists(self.pairs_path):
            with open(self.pairs_path,'rb') as f:
                self.pairs_new_lis = [pickle.load(f)]
        self.criterion = nn.CrossEntropyLoss()
        self.criterion_bce = nn.BCEWithLogitsLoss()
        self.cav_direcs = {}
        self.prefix = ""
        if self.hparams["cav_space"] == 'debiased_student':
            self.prefix = "debiased_student"

        self.fc_save_path = ''
        if self.map_type=='T_to_S':
            self.fc_save_path = DATA_PATH+f'fc_new/dino_mapped_to_{self.prefix}stu_space_num_imgs'+str(self.num_imgs)+self.dset+'/'
        else:
            self.fc_save_path = DATA_PATH+'fc_new/direct_dino_pca_space_num_imgs'+str(self.num_imgs)+self.dset+'/'
        
        if not os.path.exists(self.fc_save_path):
            os.makedirs(self.fc_save_path)
        
        

    def save_student_outs(self,pairs_vals):
        """
        Step 1 for training mapping module
        """
        img_dataset = ImgsDataset(self.model_type,self.input_shape,concept_set_type=str(pairs_vals),num_imgs = self.num_imgs,transform=self.transform)
#         img_data = ImgsDataset(self.model_type,self.input_shape,self.num_imgs,concept_set_type=str(pairs_vals))
        dataloader = DataLoader(img_dataset, batch_size=1,
                                shuffle=True, num_workers=0)
        

        save_path = f"/ssd_scratch/cvit/avani.gupta/{self.prefix}acts_"+self.model_type+self.bottleneck_name+'/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for i, (img_data, con) in enumerate(dataloader):
            img_data = img_data.squeeze(0)
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
        upconv_m = self.upconv_m
        downconv_m = self.downconv_m
        mse_loss = nn.MSELoss()
        upconv_m.train()
        downconv_m.train()
        upconv_m.cuda()
        downconv_m.cuda()
        model_params = list(upconv_m.parameters())+list(downconv_m.parameters())
        optimizer = torch.optim.Adam([p for p in model_params if p.requires_grad], lr=1e-4)
        loss_lis = []
        first = 1
        tolerance = 1
        voil_count = 0
        best_val_loss = 10e5
        for ep in tqdm(range(5)):
            for con in concepts:
                img_fea = torch.load(DATA_PATH+f'dino_fea_'+str(self.input_shape[0])+'by'+str(self.input_shape[1])+'/'+con+'all.pt').cuda()[:self.num_imgs]
                stu_pred = torch.load(f"/ssd_scratch/cvit/avani.gupta/{self.prefix}acts_"+self.model_type+self.bottleneck_name+'/'+con+'all.pt').cuda()[:self.num_imgs]
                fea_s = img_fea.shape
                for i in range(min(len(stu_pred), len(img_fea))):
                    img_ = img_fea[i]
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

                # val_loss_lis = []
                # with torch.no_grad():
                #     for con in concepts:
                #         img_fea = torch.load(DATA_PATH+'dino_fea_'+str(self.input_shape[0])+'by'+str(self.input_shape[1])+'/'+con+'all.pt').cuda()[self.num_imgs-20:]
                #         stu_pred = torch.load("/ssd_scratch/cvit/avani.gupta/acts_"+self.model_type+self.bottleneck_name+'/'+con+'all.pt').cuda()[self.num_imgs-20:]
                #         fea_s = img_fea.shape
                #         for i in range(min(len(stu_pred), len(img_fea))):
                #             img_ = img_fea[i]
                #             x = downconv_m(img_)
                #             x_= upconv_m(x)
                #             stu_pred_ = stu_pred[i].squeeze(0)
                #             assert(img_.shape==x_.shape)
                #             assert(x.shape==stu_pred_.shape)
                #             loss = mse_loss(img_, x_) + mse_loss(x, stu_pred_)
                #             assert(torch.isnan(torch.tensor(loss.item()))==False)
                #             val_loss_lis.append(loss.item())
                # val_loss_lis = np.array(val_loss_lis)
                # val_loss = np.mean(val_loss_lis)

                # if val_loss > best_val_loss:
                #     if voil_count > tolerance:
                #         break
                #     else:
                #         voil_count += 1
                # else:
                #     voil_count = 0
            print("ep",ep," loss",np.mean(loss_lis))
        
        path = DATA_PATH+'teacher_2{self.prefix}/'+self.dset+'pairs_vals'+str(self.pairs_vals)
        if not os.path.exists(path):
            os.makedirs(path)

        if self.write:
            print("saved to path")
            torch.save(upconv_m.state_dict(), path+'upconv.pt')
            torch.save(downconv_m.state_dict(), path+'downconv.pt')
        self.downconv_m = downconv_m.eval()


    def train_cavs(self):
        """
        Step 3 training CAV in teacher/student space; all other components frozen including mapping module
        """
        new_pairs = {}
        for pairs_main in self.pairs_lis:
            for main_con in pairs_main:
                pairs = pairs_main[main_con]
                for pair in pairs:
                    con, neg_con = pair
                    acts = {}
                    fc_save_path = None
                    if self.hparams["cav_space"] == 'mapped_teacher':
                        acts[con] = torch.load(DATA_PATH+'dino_fea_'+str(self.input_shape[0])+'by'+str(self.input_shape[0])+'/'+con+'all.pt')[:self.num_imgs]
                        acts[neg_con] = torch.load(DATA_PATH+'dino_fea_'+str(self.input_shape[0])+'by'+str(self.input_shape[0])+'/'+neg_con+'all.pt')[:self.num_imgs]
                        with torch.no_grad():
                            acts[con] = self.downconv_m(acts[con].cuda())
                            acts[neg_con] = self.downconv_m(acts[neg_con].cuda())
                        self.fc_save_path = DATA_PATH+f'fc/dino_mapped_to_{self.prefix}stu_space_'+self.dset+'/'

                    elif self.hparams["cav_space"] == 'student' or self.hparams["cav_space"] == 'debiased_student':
                        acts[con] = torch.load(f"/ssd_scratch/cvit/avani.gupta/{self.prefix}acts_"+self.model_type+self.bottleneck_name+'/'+con+'all.pt')[:self.num_imgs]
                        acts[neg_con] = torch.load(f"/ssd_scratch/cvit/avani.gupta/{self.prefix}acts_"+self.model_type+self.bottleneck_name+'/'+neg_con+'all.pt')[:self.num_imgs]
                        self.fc_save_path = DATA_PATH+f'fc/{self.prefix}student_space_'+self.dset+'/'
                    
                    elif self.hparams["cav_space"] == 'teacher':
                        acts[con] =  torch.load(DATA_PATH+'dino_fea_'+str(self.input_shape[0])+'by'+str(self.input_shape[1])+'/'+con+'all.pt').cuda()[:self.num_imgs]
                        acts[neg_con] =  torch.load(DATA_PATH+'dino_fea_'+str(self.input_shape[0])+'by'+str(self.input_shape[1])+'/'+neg_con+'all.pt').cuda()[:self.num_imgs]
                        self.fc_save_path = DATA_PATH+'fc/direct_dino_pca_space_'+self.dset+'/'

                    print(self.fc_save_path)
                    if not os.path.exists(self.fc_save_path):
                        os.makedirs(self.fc_save_path)

                    cav, fc_state_dict = get_or_train_cav_nn([con, neg_con], 'dino', acts)
                    direc = cav.get_direction(con) 
                    direc = torch.tensor(direc) ## take the first tensor as pos con's direc

                    if con in self.cav_direcs:
                        self.cav_direcs[con] = self.hparams["cav_wt"]*direc + (1-self.hparams["cav_wt"])*self.cav_direcs[con]
                    else:
                        self.cav_direcs[con] = direc
                    
                    if cav.accuracies['overall'] > 0.6:
                        torch.save(fc_state_dict, self.fc_save_path+''+'_cons_'.join(pair)+str(cav.accuracies["overall"])+'fc.pt')  
                        if con in new_pairs:
                            new_pairs[con].append((con,neg_con))
                        else:
                            new_pairs[con] = [pair]
                    
                with open(self.pairs_path, 'wb') as handle:
                    pickle.dump(new_pairs, handle, protocol=pickle.HIGHEST_PROTOCOL) 
                print("saved to", self.pairs_path)
                self.pairs_new_lis = [new_pairs]                 

    def update_proto(self,X_full=None, y_full=None):
        proto_dic = self.proto_dic
        if self.model_type =='resnet50': ##diff procedure for updating resnet50 proto
            for c in tqdm(self.class_labels):
                classdataset = ClassWiseImgsDataset(path=DATA_PATH+'Imagenet2012/Imagenet-sample/train/'+self.hparams["label_to_class"][c],shape=(224,224),transform=self.hparams["transforms"])
                classdataloader = torch.utils.data.DataLoader(classdataset, batch_size=4,
                                                            shuffle=False, num_workers=0)
                acts_ = []
                with torch.no_grad():
                    for cur_iter, data in enumerate(classdataloader):
                        imgs, gt = data
                        def save_activation_hook(mod, inp, out):
                            global bn_activation
                            bn_activation = out

                        handle = named_layers[bottleneck_name].register_forward_hook(save_activation_hook)
                        out = model(imgs.cuda())
                        acts_.append(bn_activation.detach().cpu())

                acts_ = torch.concat(acts_, axis=0)
                ss = acts_.shape
                acts_ = acts_.detach().cpu().numpy().reshape((-1,ss[1]*ss[2]*ss[3]))

                kmeans = KMeans(n_clusters=self.haparams['knn_k'], random_state=0).fit(acts_)
                centers = kmeans.cluster_centers_

                for f in range(self.hparams['knn_k']):
                    if epoch==0 or not hparams['do_proto_mean']:
                        if c in proto_dic:
                            proto_dic[c]['cluster_'+str(f)] = centers[f].reshape(ss[1],ss[2],ss[3])
                        else:
                            proto_dic[c] = {}
                            proto_dic[c]['cluster_'+str(f)] = centers[f].reshape(ss[1],ss[2],ss[3])
                    else:
                        proto_dic[c]['cluster_'+str(f)] = (1-hparams["cur_proto_mean_wt"])*proto_dic[c]['cluster_'+str(f)]+hparams["cur_proto_mean_wt"]*centers[f].reshape((ss[1],ss[2],ss[3]))
            self.proto_dic = proto_dic
        else:
            X_full, y_full = X_full.cuda(), y_full.cuda()
            # X_full = torch.cat([x for x, y in batches])
            # y_full = torch.cat([y for x, y in batches])
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

    def cdep_loss(self, imgs, gt, gt_loss):
        if self.model_type =='colormnist' or self.model_type=='texturemnist' or self.model_type =='cat_dog' or self.model_type =='faces':
            num_blobs = 8
            regularizer_rate = self.hparams["regularizer_rate"]
            blobs = np.zeros((28*28,28,28))
            for i in range(28):
                for j in range(28):
                    blobs[i*28+j, i, j] =1
            imgs = imgs.cuda()
            add_loss = torch.zeros(1,).cuda()
            blob_idxs = np.random.choice(28*28, size = num_blobs, p = prob)
            if self.hparams['cdep_grad_method'] ==0:
                for i in range(num_blobs): 
                    add_loss += score_funcs.cdep(model, imgs, blobs[blob_idxs[i]],model_type = 'mnist')
                (regularizer_rate*add_loss+gt_loss).backward(retain_graph=True)
            elif self.hparams['cdep_grad_method'] ==1:
                for i in range(num_blobs): 
                    add_loss +=score_funcs.gradient_sum(imgs, gt.to(self.device), torch.FloatTensor(blobs[blob_idxs[i]]).to(self.device),  model, torch.nn.functional.nll_loss)
                (regularizer_rate*add_loss).backward(retain_graph=True)
                # loss = torch.nn.functional.nll_loss(out, gt.to(self.device))
                # loss.backward(retain_graph=True)
            elif self.hparams['cdep_grad_method'] ==2:
                for j in range(len(imgs)):
                    for i in range(num_blobs): 
                        add_loss +=(score_funcs.eg_scores_2d(model, imgs, j, gt.to(self.device), 50) * torch.FloatTensor(blobs[blob_idxs[i]]).to(self.device)).sum()

                (regularizer_rate*add_loss).backward(retain_graph=True)
                
        elif self.model_type =='decoymnist':
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
                if self.hparams['cdep_grad_method'] ==0:
                    rel, irrel = cdep.cd(blob, imgs,model)
                    add_loss += torch.nn.functional.softmax(torch.stack((rel.view(-1),irrel.view(-1)), dim =1), dim = 1)[:,0].mean()
                    (regularizer_rate*add_loss +gt_loss).backward(retain_graph=True)
                elif self.hparams['cdep_grad_method'] ==1:
                    add_loss +=score_funcs.gradient_sum(imgs, gt.to(self.device), torch.FloatTensor(blob).to(self.device),  model, torch.nn.functional.nll_loss)
                    (regularizer_rate*add_loss).backward(retain_graph=True)

                elif self.hparams['cdep_grad_method'] ==2:
                    for j in range(len(imgs)):
                        add_loss +=(score_funcs.eg_scores_2d(model, imgs, j, gt.to(self.device), num_samples) * torch.FloatTensor(blob).to(self.device)).sum()
                    (regularizer_rate*add_loss).backward(retain_graph=True)
        return regularizer_rate*add_loss

    def update(self, minibatches, unlabeled=None):
        running_loss = 0
        gt_loss = 0
        concept_loss = 0
        x, y = minibatches
        x, y = x.cuda(), y.cuda()
        self.optimizer.zero_grad()
        gt_loss = F.cross_entropy(self.predict(x), y)
        gt_loss.backward(retain_graph=True)
        if self.hparams['use_cdepcolor']:
            cdep_loss = self.cdep_loss(x, y, gt_loss)
            running_loss += cdep_loss.item()
        running_loss += gt_loss.item()
        gt_loss += gt_loss.item()
        for pn,pairs_all in enumerate(self.pairs_new_lis):
            for main_con in pairs_all:
                pairs = pairs_all[main_con]
                for qs,pair in enumerate(pairs):
                    affect = 0 #self.affect_lis[qs]
                    c_loss = self.get_concept_loss(x,y,pair,affect)
                    concept_loss += c_loss.item()
                    if pn==len(self.pairs_lis)-1:
                        c_loss.backward()
                        running_loss+= c_loss.item()
                    else:
                        c_loss.backward(retain_graph=True)
                        running_loss+= c_loss.item()
        self.optimizer.step()
        return {'loss': running_loss, "concept_loss":concept_loss, "gt_loss":gt_loss}
    
    def eval_model(self, minibatches,split='val'):
        self.network.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.cuda(), target.cuda()
                if model_type=='faces': #reduce one hot encoding to binry (for faces its just binary classi so take first col as labels)
                    target = target[:,0]
                output = self.predict(data)
                if model_type == 'faces':
                    val_loss += self.criterion(output, target).item()
                elif model_type=='cat_dog':
                    val_loss += self.criterion_bce(output, target.view(len(target), 1).to(device)).item()
                else:
                    val_loss += torch.nn.functional.nll_loss(output, target, reduction='sum').item() # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc = 100. * correct / len(val_loader.dataset)
        self.network.train()
        return {split+"_loss":val_loss, split+"_acc":val_acc}
    
    def load_pre_calc_cavs(self, pair):
        files =  glob.glob(self.fc_save_path+'_cons_'.join(pair)+'*.pt')
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

        self.cav_direcs[pair[0]] = torch.tensor(cavs[0])
        self.cav_direcs[pair[0]].requires_grad = True

    def get_concept_loss(self,imgs,gt,pair,affect): 
        named_layers = dict(self.network.named_modules())            
        
        if pair[0] not in self.cav_direcs:
            self.load_pre_calc_cavs(pair)

        bn_activation= None
        dot_lis = []
        cos_lis = []
        mse_dvec_lis = []
        self.device = 'cuda:0'
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
            # if self.map_type=='S_to_T':
            #     act = self.downconv_m(act)

            if self.hparams["use_proto"]:
                for f in range(self.hparams["knn_k"]):
                    if self.hparams['model_type'] == 'resnet50':
                        lb = self.hparams["label_to_class"][gt[b].item()]
                        if f==0:
                            loss_a = self.mse_loss(act, torch.from_numpy(self.proto_dic[lb]['cluster_'+str(f)]).cuda().unsqueeze(0))
                        else:
                            loss_a += self.mse_loss(act, torch.from_numpy(self.proto_dic[lb]['cluster_'+str(f)]).cuda().unsqueeze(0))
                    else:
                        if f==0:
                            loss_a = self.mse_loss(act, torch.from_numpy(self.proto_dic[gt[b].item()]['cluster_'+str(f)]).cuda().unsqueeze(0))
                        else:
                            loss_a += self.mse_loss(act, torch.from_numpy(self.proto_dic[gt[b].item()]['cluster_'+str(f)]).cuda().unsqueeze(0))
                grad_ = torch.autograd.grad(loss_a, act, retain_graph=True,create_graph=True)
                
            else:
                grad_ = torch.autograd.grad(out[0][gt[b]], act, retain_graph=True,create_graph=True)
           
            dot = torch.dot(grad_[0].to(self.device).float().squeeze(0).flatten(),self.cav_direcs[pair[0]].to(self.device).float())/torch.linalg.norm(self.cav_direcs[pair[0]])
            dot_lis.append(dot)
            
            unit_grad = grad_[0].to(self.device).float().squeeze(0).flatten()
            unit_direc = self.cav_direcs[pair[0]].to(self.device).float()/torch.linalg.norm(self.cav_direcs[pair[0]])
            unit_grad = unit_grad/torch.linalg.norm(unit_grad)
            mse_dvec_lis.append(self.mse_loss(unit_grad, unit_direc))

            cos_ = self.cos(grad_[0].to(self.device).float().squeeze(0).flatten(),self.cav_direcs[pair[0]].to(self.device).float())
            cos_lis.append(cos_)

            handle.remove()

        dot_lis = torch.stack(dot_lis)
        mse_dvec_lis = torch.stack(mse_dvec_lis)
        cos_lis = torch.stack(cos_lis)
        
        if affect==False:
            loss_ = torch.sum(torch.abs(cos_lis)) #L1
        if affect==True:
            loss_= self.mse_loss(cos_lis, torch.ones(len(cos_lis)).to(self.device))
        return loss_


    def predict(self, x):
        return self.network(x)

from argparse import ArgumentParser
import sys
parser = ArgumentParser()
parser.add_argument("--knn_k", default=7, type=int, help="knn value")
parser.add_argument("--cur_proto_mean_wt", default=0.3, type=float)
parser.add_argument("--update_freq", default=500, type=int)
parser.add_argument("--proto_update_prob", default=0.2, type=float)
parser.add_argument("--use_proto", default=True, type=bool)
parser.add_argument("--do_proto_mean", default=True, type=bool)
parser.add_argument("--update_cavs", default=True, type=bool)
# parser.add_argument("--model_type", default='resnet50')
# parser.add_argument("--dset", default='imgnet')
# parser.add_argument("--pairs_vals", default="textures", type=str)
# parser.add_argument("--bottleneck_name", default='layer4.2.conv3', type=str)
parser.add_argument("--model_type", default='colormnist')
parser.add_argument("--dset", default='mnist')
parser.add_argument("--pairs_vals", default="8", type=str)
parser.add_argument("--bottleneck_name", default='conv2', type=str)
parser.add_argument("--num_imgs", default=150, type=int)
parser.add_argument("--lr", default=0.1, type=float)
parser.add_argument("--cav_wt", default=0.3, type=float)
parser.add_argument("--affect_lis", default=0, type=int)
parser.add_argument("--train_from_scratch", default=0, type=int)
parser.add_argument('--regularizer_rate', type=float, default=0.3, help='regularizer rate for cdep')
parser.add_argument('--use_cdepcolor', type=int, default=1, help='whether to use cdep color')
parser.add_argument('--teach_mapped_to_stu', type=int, default=1)
parser.add_argument('--cdep_grad_method', type=int, default=1, help='grad method cdep')
parser.add_argument('--cav_space', type=str, default='student', help='cav_space: teacher, student, mapped_teacher, debiased_student')
parser.add_argument('--checkpoints_dir', type=str, default=DATA_PATH+'new_checkpoints/', help='models are saved here')
# sys.argv = ['']
# parser.set_defaults(verbose=False)
hparams = parser.parse_args()
# from da import *
from utils.imdb_classi_test import *
from utils.networks_classi import *

model_type = hparams.model_type
num_imgs = hparams.num_imgs
batch_size = 64
# wandb.init(project="change_cav_new", entity="avani",config=hparams, save_code=True)
hparams = vars(hparams)

if  model_type == 'resnet50':
    val_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    shape = (224,224)
    model = resnet50(pretrained=True)
    model = model.cuda()
    bs = 32
    imagenet_zip_path = DATA_PATH+'Imagenet2012/Imagenet-sample/'
    with open('imagenet_class_index.json', "r") as f:
        data = json.load(f)
    label_mapping = {}
    for key, value in data.items():
        label_mapping[value[0]] = int(key)
    label_to_class = {value: key for key, value in label_mapping.items()}
    hparams["label_mapping"] = label_mapping
    hparams["label_to_class"] = label_to_class

    train_dataset =  AllClassesImgsDataset(path=imagenet_zip_path+'train/',shape=(224,224),label_mapping=label_mapping, transform=val_transforms)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs,
                                            shuffle=True, num_workers=2)

    val_dataset =  AllClassesImgsDataset(path=imagenet_zip_path+'val/',shape=(224,224),label_mapping=label_mapping, transform=val_transforms)
    
    
    val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs,
                                            shuffle=True, num_workers=2)

    ### make sure loaded labels match with pre-trained model's sense of labels
    assert(all(train_dataset.class_to_idx[key] == label_mapping[key] for key in train_dataset.class_to_idx))
    assert(all(val_dataset.class_to_idx[key] == label_mapping[key] for key in val_dataset.class_to_idx))

    test_loader = None
    named_layers = dict(model.named_modules())
    lis = list(named_layers.keys())
    bottleneck_name = "layer4.2.conv3"
    class_names = train_dataset.class_to_idx.keys()
    class_names_to_labels_dic = train_dataset.class_to_idx
    class_labels = list(train_dataset.class_to_idx.values())
    hparams["transforms"] = val_transforms

if model_type == 'faces':
    model = resnet18(pretrained=False)
    model.fc = nn.Linear(512,2)
    if not hparams["train_from_scratch"]:
        model.load_state_dict(torch.load("/ssd_scratch/cvit/avani.gupta/bffhq/bffhq/bffhq_0.5_vanilla/result/best_model.th")['state_dict'])
    model.cuda()
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
    class_labels = [0,1]
    hparams["transforms"] = val_transforms

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
    class_labels = [0,1]


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
    if not hparams["train_from_scratch"]:
        model.load_state_dict(torch.load('mnist/DecoyMNIST/orig_model_decoyMNIST_.pt'))
    bottleneck_name = 'conv2'
    num_classes = 10
    full_X_proto = True
    class_labels = np.arange(0,10)


if model_type =='colormnist':
    x_numpy_train = np.load(os.path.join("dep/data/ColorMNIST", "train_x.npy"))
    prob = (x_numpy_train.sum(axis = 1) > 0.0).mean(axis = 0).reshape(-1)
    prob /=prob.sum()
    mean = x_numpy_train.mean(axis = (0,2,3))
    std = x_numpy_train.std(axis = (0,2,3))
    bottleneck_name = hparams["bottleneck_name"]
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
    if not hparams["train_from_scratch"]:
        if hparams['cav_space']=='debiased_student':
            model.load_state_dict(torch.load(DATA_PATH+'acc46.03best_val_acc45.28highest_epoch0iter0colormnistmnistp8nimg150lr0.01rr0.3wtcav5bs44pwt0.3upr1cd1precalc1up1scratch0uknn1cwt0s42corr'))
        else:
            model.load_state_dict(torch.load('mnist/ColorMNIST/orig_model_colorMNIST.pt'))

    num_classes = 10
    full_X_proto = True
    class_labels = np.arange(0,10)
    
if model_type =='texturemnist':
    x_numpy_train = np.load(os.path.join("/ssd_scratch/cvit/avani.gupta/data/ColorMNIST", "train_texture_x.npy"))
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
    class_labels = np.arange(0,10)
    if not hparams["train_from_scratch"]:
        model.load_state_dict(torch.load('/home/avani.gupta/tcav_pt/methodvanilla_tex.pt'))


if not os.path.exists(hparams['checkpoints_dir']):
    os.makedirs(hparams['checkpoints_dir'])

def get_model_save_name(hparams):
    model_save_name = hparams["model_type"]+hparams["dset"]
    k = ["pairs_vals","num_imgs","lr","regularizer_rate","wtcav","batch_size","cur_proto_mean_wt","use_proto","use_cdepcolor","use_precalc_proto","update_proto","train_from_scratch","use_knn_proto","class_wise_training"]
    ks = {"pairs_vals":"p","num_imgs":"nimg","lr":"lr","regularizer_rate":"rr","wtcav":"wtcav", "batch_size":"bs","cur_proto_mean_wt":"pwt","use_proto":"upr","use_cdepcolor":"cd","use_precalc_proto":"precalc","update_proto":"up","train_from_scratch":"scratch","use_knn_proto":"uknn","class_wise_training":"cwt"}
    for p in k:
        if p in hparams:
            if isinstance(hparams[p],str):
                model_save_name += ks[p]+str(hparams[p])
            elif isinstance(hparams[p],(int, float)):
                model_save_name += ks[p]+str(round(hparams[p],2))
            else:
                model_save_name += ks[p]+str(hparams[p])
    print("model save name",model_save_name)
    return model_save_name

hparams['model_save_name']  = get_model_save_name(hparams)
custom_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x / 255 if torch.max(x) > 1 else x)
])
if 'mnist' in hparams['model_type']:
    hparams["transforms"] = custom_transform
# cd = CD(shape, num_classes,hparams=hparams,network=model)#full_X_proto=full_X_proto)
cd = CD(shape, hparams=hparams,network=model, class_labels=class_labels,transform=hparams["transforms"])#full_X_proto=full_X_proto)
cd.save_student_outs(hparams["pairs_vals"])
cd.map_activation_spaces()
cd.train_cavs()
# cd.update_proto(X_full,y_full)

best_val_acc = -1
corr_test_acc = 0
epochs = 50
for epoch in tqdm(range(epochs)):
    cav_loss_lis = []
    # for cur_iter, data in enumerate(train_loader):
    for iter, data in tqdm(enumerate(train_loader)):
        metrics_dic = cd.update(data)
        if iter!=0 and iter%hparams["update_freq"]==0:
            cd.save_student_outs(hparams["pairs_vals"])
            cd.map_activation_spaces()
            cd.train_cavs()
            if hparams["model_type"]=='resnet50':
                cd.update_proto()
            else:
                cd.update_proto(X_full,y_full)
        if iter%10 != 0: ##log train metrics in each iter
            wandb.log(metrics_dic)
        if iter%10==0:
            val_metrics = cd.eval_model(val_loader)
            val_acc = val_metrics['val_acc']
            metrics_dic.update(val_metrics)
            if test_loader:
                test_metrics = cd.eval_model(test_loader,split="test")
                metrics_dic.update(test_metrics)
            wandb.log(metrics_dic)
            if val_acc > best_val_acc:
                wandb.run.summary[f"best_val_acc_"] = val_acc
                if test_loader:
                    corr_test_acc = metrics_dic["test_acc"]
                    wandb.run.summary[f"best_val_corr_test_acc"] = metrics_dic["test_acc"]
                wandb.run.summary[f"best_val_epoch"] = epoch
                wandb.run.summary[f"best_val_iter"] = iter
                wandb.run.summary[f"best_val_acc_cpk"] = hparams["model_save_name"]+"corr"
                torch.save(model.state_dict(), os.path.join(hparams["checkpoints_dir"],"best_val_acc"+str(round(val_acc,2))+"highest_epoch"+str(epoch)+"iter"+str(iter)+hparams["model_save_name"]+"corr"))
                best_val_acc = val_acc
