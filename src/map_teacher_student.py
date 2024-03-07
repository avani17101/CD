from turtle import down
import torch
import torch.nn as nn
import numpy as np
from statistics import mean
from utils.utils_train import *
#### cav module
from utils.utils_tcav2 import *
import torch
import torch.nn as nn
# from teacher import calc_params
import numpy as np
import os.path
import pickle
from six.moves import range
from tqdm import tqdm
from argparse import ArgumentParser



if __name__ == "__main__":
    direcs = {}
    parser = ArgumentParser()
    parser.add_argument("--num_imgs", default=150, type=int, help="number of imgs total")
    parser.add_argument("--pairs_vals", default=5, type=int, help="pair number")
    parser.add_argument("--epochs", default=5, type=int, help="epochs")
    parser.add_argument("--model_type", default="decoymnist", help="pair number")
    parser.add_argument("--dset", default="dmnist", help="pair number")
    parser.add_argument("--teacher_type", default="dino")

    parser.set_defaults(verbose=False)
    opt = parser.parse_args()
    model_type = opt.model_type
    pairs_vals = opt.pairs_vals
    teacher_type = opt.teacher_type
    upconv_m, downconv_m, dset, bottleneck_name = get_mapping_module(model_type)

    pairs = get_pairs(model_type,str(pairs_vals))
    concepts = set()
    for p in pairs:
        concepts.add(p)
        for l1,l2 in pairs[p]:
            concepts.add(l1)
            concepts.add(l2)

    concepts = list(concepts)
    # concepts = list(pairs.keys())
    # print(concepts)
    save_path = "/media/Data2/avani.gupta/acts_"+model_type+bottleneck_name+'/'
    ex = torch.load(save_path+concepts[0]+'all.pt')
    mse_loss = nn.MSELoss()
    upconv_m.train()
    downconv_m.train()
    upconv_m.cuda()
    downconv_m.cuda()

    model_params = list(upconv_m.parameters())+list(downconv_m.parameters())
    # optimizer = torch.optim.SGD([p for p in model_params if p.requires_grad],lr=learning_rate)
    optimizer = torch.optim.Adam([p for p in model_params if p.requires_grad], lr=1e-4)
    loss_lis = []
    first = 1
    # print("here")
    for ep in tqdm(range(opt.epochs)):
        for con in tqdm(concepts):
            # try:
                if model_type =='toy_conv':
                    img_fea = torch.load('/media/Data2/avani.gupta/'+teacher_type+'_fea_200by200/'+con+'all.pt').cuda()
                elif model_type =='colormnist' or model_type=='decoymnist' or model_type=='texturemnist':
                    img_fea = torch.load('/media/Data2/avani.gupta/'+teacher_type+'_fea_28by28/'+con+'all.pt').cuda()
                elif model_type=='clever_hans' or model_type =='clever_hans7' or model_type=='cat_dog' or model_type=='faces' or model_type=='pacs':
                    img_fea = torch.load('/media/Data2/avani.gupta/'+teacher_type+'_fea_224by224/'+con+'all.pt').cuda()
                elif model_type =='isic' or model_type =='isic_vgg':
                    img_fea = torch.load('/media/Data2/avani.gupta/'+teacher_type+'_fea_450by600/'+con+'all.pt').cuda()
                else:
                    img_fea = torch.load('/media/Data2/avani.gupta/'+teacher_type+'_fea/'+con+'.pt').cuda()
                # breakpoint()
                # print("/media/Data2/avani.gupta/acts_"+model_type+bottleneck_name+'/'+concepts[0]+'all.pt')
                print("/media/Data2/avani.gupta/acts_"+model_type+bottleneck_name+'/'+con+'all.pt')
                stu_pred = torch.load("/media/Data2/avani.gupta/acts_"+model_type+bottleneck_name+'/'+con+'all.pt').cuda()
                # breakpoint()
                fea_s = img_fea.shape
                
    
                # print(fea_s)
                
                if model_type =='isic':
                    img_fea = img_fea.reshape(-1,268800)
                if model_type =='toy':
                    img_fea = img_fea.reshape(-1,40000)
                img_fea = img_fea[:opt.num_imgs]
                for i,img_ in enumerate(img_fea):
                    x = downconv_m(img_)
                    # import pdb
                    # pdb.set_trace()
                    x_= upconv_m(x)
                    # print(img_.shape, x_.shape ,x.shape, stu_pred.shape)
                    if model_type=='cat_dog' or model_type=='faces' or model_type=='pacs':
                        stu_pred_ = stu_pred[i].squeeze(0)
                    else:
                        stu_pred_ = stu_pred[i]
                    assert(img_.shape==x_.shape)
                    assert(x.shape==stu_pred_.shape)
                    
                    loss = mse_loss(img_, x_) + mse_loss(x, stu_pred_)
                    # print(loss)
                    if i!=len(img_fea)-1:
                        loss.backward(retain_graph=True)
                    else:
                        loss.backward()
                    assert(torch.isnan(torch.tensor(loss.item()))==False)
                    loss_lis.append(loss.item())
                        
                optimizer.step()
                optimizer.zero_grad()
            # except Exception as e:
            #     print(e,"error in ",con)
            #     breakpoint()

        print("ep",ep," loss",mean(loss_lis))

        path = '/media/Data2/avani.gupta/'+opt.teacher_type+'/'+opt.dset+'pairs_vals'+str(pairs_vals)+"num_imgs"+str(opt.num_imgs)+"ep"+str(opt.epochs)
        if not os.path.exists(path):
            os.makedirs(path)
        print(path)

        torch.save(upconv_m.state_dict(), path+'upconv.pt')
        torch.save(downconv_m.state_dict(), path+'downconv.pt')

