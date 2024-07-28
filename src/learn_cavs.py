'''
Learn fc over dino mapped to student space
'''
import torch
import os.path
import pickle
import os
from utils.utils_train import *
from utils.networks_classi import *
from argparse import ArgumentParser

"""
Teacher learns fc(tcav) on concept set
"""
teach_mapped_to_stu = True #set false if train on original dino-pca features
from utils.utils_train import *
from utils.utils_tcav2 import *

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--num_imgs", default=150, type=int, help="number of imgs total")
    parser.add_argument("--pairs_vals", default=5, type=int, help="pair number")
    parser.add_argument("--mapping_mod_epoch", default=5, type=int, help="whuch ep mapping module to load")
    parser.add_argument("--model_type", default="decoymnist", help="pair number")
    parser.add_argument("--dset", default="dmnist", help="pair number")
    parser.add_argument("--teacher_type", default="dino")


    parser.set_defaults(verbose=False)
    opt = parser.parse_args()

    # model_type = 'colormnist'
    # dset = 'mnist'
    model_type = opt.model_type
    dset = opt.dset
    teacher_type = opt.teacher_type
    if model_type == 'decoymnist':
      dset = 'dmnist'
    if model_type =='toy':
      dset = 'toy'
    if model_type =='clever_hans':
      dset = 'clever_hans'
    if model_type =='clever_hans7':
      dset = 'clever_hans7'
    if model_type =='isic_vgg':
      dset = 'isic_vgg'
    pairs_vals = opt.pairs_vals
    num_imgs = opt.num_imgs
    pairs_main = get_pairs(model_type, str(pairs_vals))
    print(pairs_main)
    path = '/media/Data2/avani.gupta/'
    new_pairs = {}
    upconv_m, downconv_m, dset, bottleneck_name = get_mapping_module(opt.model_type)

    

    #load mapping model
    if teach_mapped_to_stu:
      path ='/media/Data2/avani.gupta/'+teacher_type+'/'+dset+'pairs_vals'+str(pairs_vals)+'num_imgs'+str(num_imgs)+"ep"+str(opt.mapping_mod_epoch)
      downconv_m.load_state_dict(torch.load(path+'downconv.pt'))
      downconv_m.eval()

    for main_con in pairs_main:
      pairs = pairs_main[main_con]
      for pair in pairs:
          con, neg_con = pair
          
          acts = {}
          if model_type =='isic_vgg' or model_type =='isic':
            acts[con] = torch.load('/media/Data2/avani.gupta/'+teacher_type+'_fea_450by600/'+con+'all.pt')[:num_imgs]
            acts[neg_con] = torch.load('/media/Data2/avani.gupta/'+teacher_type+'_fea_450by600/'+neg_con+'all.pt')[:num_imgs]
          elif model_type =='colormnist' or model_type =='decoymnist' or model_type=='texturemnist':
            acts[con] = torch.load('/media/Data2/avani.gupta/'+teacher_type+'_fea_28by28/'+con+'all.pt')[:num_imgs]
            acts[neg_con] = torch.load('/media/Data2/avani.gupta/'+teacher_type+'_fea_28by28/'+neg_con+'all.pt')[:num_imgs]
          elif model_type=='cat_dog' or model_type =='faces' or model_type=='pacs':
            acts[con] = torch.load('/media/Data2/avani.gupta/'+teacher_type+'_fea_224by224/'+con+'all.pt')[:num_imgs]
            acts[neg_con] = torch.load('/media/Data2/avani.gupta/'+teacher_type+'_fea_224by224/'+neg_con+'all.pt')[:num_imgs]
          else:
            acts[con] = torch.load('/media/Data2/avani.gupta/'+teacher_type+'_fea/'+con+'all.pt')[:num_imgs]
            acts[neg_con] = torch.load('/media/Data2/avani.gupta/'+teacher_type+'_fea/'+neg_con+'all.pt')[:num_imgs]

          if teach_mapped_to_stu:
            fea_s = acts[con].shape
            with torch.no_grad():
              if model_type == 'toy':
                acts[con] = acts[con].reshape(-1, 40000)
                acts[neg_con] = acts[neg_con].reshape(-1, 40000)
              
              acts[con] = downconv_m(acts[con])
              acts[neg_con] = downconv_m(acts[neg_con])
          # print(acts[con].shape, acts[neg_con].shape)
          # breakpoint()

          cav, fc_state_dict = get_or_train_cav_nn([con, neg_con], 'dino', acts)
          
          fc_save_path = ''
          # if teach_mapped_to_stu:
          fc_save_path = '/media/Data2/avani.gupta/fc_new/'+teacher_type+'_mapped_to_stu_space_num_imgs'+str(num_imgs)+dset+"M_ep"+str(opt.mapping_mod_epoch)+'/'
          # else:
          #   fc_save_path = '/media/Data2/avani.gupta/fc_new/direct_dino_pca_space_num_imgs'+str(num_imgs)+dset+'/'
          
          if not os.path.exists(fc_save_path):
              os.makedirs(fc_save_path)

          if cav.accuracies['overall'] > 0.7:
              torch.save(fc_state_dict, fc_save_path+''+'_cons_'.join(pair)+str(cav.accuracies["overall"])+'fc.pt')  
              if con in new_pairs:
                new_pairs[con].append((con,neg_con))
              else:
                new_pairs[con] = [pair]
        
      print(new_pairs)
    #map1 : large mapping 
    print('pairs_neww/pairs'+str(pairs_vals)+'_large_'+teacher_type+'_mapped_to_stu_space_imgs'+str(num_imgs)+dset+"M_ep"+str(opt.mapping_mod_epoch)+'.pickle')
    with open('pairs_neww/pairs'+str(pairs_vals)+'_large_'+teacher_type+'_mapped_to_stu_space_imgs'+str(num_imgs)+dset+"M_ep"+str(opt.mapping_mod_epoch)+'.pickle', 'wb') as handle:
        pickle.dump(new_pairs, handle, protocol=pickle.HIGHEST_PROTOCOL)      