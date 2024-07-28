import os
import glob
import pickle
import json
import random
import shutil
import numpy as np
import matplotlib.pyplot as plt
import cv2
import yaml
from tqdm import tqdm
from io import StringIO 
from sklearn import linear_model, svm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from six.moves import range
from torch import nn, optim
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image

import numpy as np # linear algebra
import pandas as pd
from torchvision import datasets, models, transforms


DATA_PATH = '/media/Data2/avani.gupta/'
class IMDBTestDataSet(Dataset):
    def __init__(self, transforms):
        
        self.data_dir = DATA_PATH+'imdb_crop/'
        path = '/home/avani.gupta/tcav_pt/learning-not-to-learn/dataset/IMDB/IMDB_with_bias/'
        img_df = pd.read_csv(path+'test_list.csv')
        self.image_list = img_df['img_name'].values
        self.labels = img_df['gender'].values
        self.transforms = transforms
        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self,idx):
        path = self.data_dir+self.image_list[idx]
        lb = self.labels[idx]
        image = plt.imread(path)
        if len(image.shape) == 2: #2d img
            image = np.expand_dims(image,axis=0)
            image = torch.from_numpy(np.concatenate((image,image,image))).float()
        else: #3ch img
            image = torch.from_numpy(np.transpose(plt.imread(path),(2,0,1))/255.0).float()
        if self.transforms:
            image = self.transforms(image)
        return image, torch.tensor(lb).float()


class CatTestDataSet(Dataset):
    def __init__(self, transforms):
        self.data_dir = DATA_PATH+'dog_cats/test/'
        self.image_list = os.listdir(self.data_dir)
        self.transforms = transforms
        
    def __len__(self):
        return len(self.image_list)

    
    def __getitem__(self,idx):
        path = self.data_dir+self.image_list[idx]
        # print(path)
        image = torch.from_numpy(np.transpose(plt.imread(path),(2,0,1))/255.0).float()
        # print(image.max(),image.min())
        if self.transforms:
            image = self.transforms(image)
        return path.split('/')[-1].replace('.jpg',''),image

# from da import IMDBBiasedDataSet,  CatBiasedDataSet

if __name__ == '__main__':
    import csv
    fields = ['name', 'acc']
    rows = []

    bs = 512
    for dset in ['cat_dog']:
        for bias in ['1','2']:
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

            # bias = 'TB1'
            # dset = 'cat_dog'
            test_on_biased_other = True
            data_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                ])
            if test_on_biased_other:
                if dset == "cat_dog":
                    ds = CatBiasedDataSet(data_transform,opp_bias)
                else:
                    ds = IMDBBiasedDataSet(data_transform,opp_bias)

            test_dl = DataLoader(ds, bs)
            model = models.resnet18(pretrained=True)
            
            in_feats = model.fc.in_features
            model.fc = nn.Linear(in_feats, 1)
            for n in glob.glob(DATA_PATH+''+dset+'_model_transfer_learn'+bias+'*.pt'):
                try:
                    model.load_state_dict(torch.load(n))
                    model = model.cuda()
                    model.eval()
                    if test_on_biased_other:
                        criterion = nn.BCEWithLogitsLoss()
                        val_total = 0
                        val_correct = 0
                        val_running_loss = 0.0
                        with torch.no_grad():
                            for i, (val_images, val_labels) in tqdm(enumerate(test_dl)):
                                val_logits = model(val_images.cuda())
                                val_labels = val_labels.view(len(val_labels), 1).cuda()
                                predicted = torch.sigmoid(val_logits)>0.5
                                val_correct += (predicted == val_labels).sum().item()
                                val_total += val_labels.shape[0]
                        acc = val_correct/val_total
                        print(n,acc)
                        rows.append([n,acc])
                except Exception as e:
                    print(e,n)

    import time
    timestr = time.strftime("%Y%m%d-%H%M%S")
    with open('imdb_direct_'+timestr+'.csv', 'w') as csvfile: 
        csvwriter = csv.writer(csvfile) 
        csvwriter.writerow(fields) 
        csvwriter.writerows(rows)

                # if dset == 'cat_dog' and not test_on_biased_other:
                #     criterion = nn.BCEWithLogitsLoss()
                #     val_total = 0
                #     val_correct = 0
                #     val_running_loss = 0.0
                #     preds = []
                #     img_paths = []
                #     with torch.no_grad():
                #         for i, (paths, val_images) in enumerate(test_dl):
                #             val_logits = torch.sigmoid(model(val_images.cuda()))
                #             preds.extend(val_logits.flatten())
                #             img_paths.extend(paths)
                #     preds = torch.stack(preds)
                #     preds = preds.cpu().numpy()
                #     dic = {'id':img_paths, 'label':preds}
                #     df= pd.DataFrame(dic)
                #     df.to_csv('tb1_orig_pred.csv',index=False)



class ClassWiseImgsDataset(Dataset):
    def __init__(self,path,shape,num_imgs=1000,transform=None):
        self.path = path
        self.shape = shape
        self.num_imgs = num_imgs
        self.files = os.listdir(self.path)
        self.transform = transform
        

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        img = plt.imread(os.path.join(self.path, file))
        img = cv2.resize(img, self.shape)
        if len(img.shape)==3:
            img = np.transpose(img,(2,0,1))
        else:
            img = np.repeat(img[np.newaxis, :,:],3, axis=0)           
        if self.transform:
            img = self.transform(torch.from_numpy(img).float())
        return img, os.path.basename(file)

class AllClassesImgsDataset(Dataset):
    def __init__(self,path,shape,label_mapping,transform=None):
        self.path = path
        self.shape = shape
        self.classes = os.listdir(self.path)
        self.files = []
        self.class_lbs = []
        self.class_to_idx = {}
        for c in self.classes:
            class_files = glob.glob(os.path.join(self.path,c)+'/*')
            self.files.extend(class_files)
            self.class_lbs.extend([label_mapping[c]]*len(class_files))
            self.class_to_idx[c] = label_mapping[c]
        self.transform = transform
        self.label_mapping = label_mapping
        

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        img = plt.imread(os.path.join(self.path, file))
        img = cv2.resize(img, self.shape)
        if len(img.shape)<3:
            img = np.repeat(img[:,:, np.newaxis],3, axis=2)           
        if self.transform:
            img = self.transform(img)
        return img, self.class_lbs[idx]

def load_image_from_file(filename, shape=(384,512),convert_srgb=False):
    # ensure image has no transparency channel
    try:
      # img = np.array(PIL.Image.open(open(filename, 'rb')).convert('RGB')).resize(shape, PIL.Image.BILINEAR)[:,:,:3]
      img = plt.imread(filename)
      img = cv2.resize(img, shape)[:,:,:3]
      # Normalize pixel values to between 0 and 1.
      if img.max()>10:
        img = np.float32(img)/ 255.0
      if convert_srgb:
          return srgb_to_rgb(img)
      else:
          return img
    except Exception as e:
        print("eror in reading img", filename)
        return []
      
def split_concept(con, path=DATA_PATH+'IID_data/concepts/'):
    '''
    splits concept folder
    '''
    c = 0
    cur = 0
    lis = os.listdir(path+con)
    random.shuffle(lis)
    for f in lis:
        if not os.path.exists(path+con+str(c)+'/'):
            os.makedirs(path+con+str(c)+'/')
        shutil.copy2(path+con+'/'+f,path+con+str(c)+'/'+f)
        cur += 1
        if cur == 200:
            c += 1
            cur = 0
        if c == 10:
            break
def get_bt_name(model_type):
    if model_type=='pacs':
        return 'layer4.2.conv3'
    if 'mnist' in model_type:
        return 'conv2'

def get_affect_lis(model_type,concept_lis):
    # if model_type=='pacs':
    return np.zeros(len(concept_lis))

def get_pairs(model_type,concept_set_type=None):
    if model_type =='resnet50':
        if concept_set_type == 'textures_train':
            pairs = {"textures_train":[("textures_train","random_discovery_imagenet_train")]}
        
        if concept_set_type == 'textures':
            pairs = {"textures":[("textures","random_discovery_imagenet")]}

    elif model_type =='pacs':
        if concept_set_type =='0':
            pairs = {"paintings":[("paintings","randoms_da"),("paintings","randoms_da2")],"sketches":[("sketches","randoms_da"),("sketches","randoms_da2")],"cartoons":[("cartoons","randoms_da"),("cartoons","randoms_da2")]}
        if concept_set_type =='1':
            pairs = {"art_painting":[("art_painting","randoms_da"),("art_painting","randoms_da2")],"sketch":[("sketch","randoms_da"),("sketch","randoms_da2")],"cartoon":[("cartoon","randoms_da"),("cartoon","randoms_da2")]}
           
    elif model_type == 'faces':
        if concept_set_type=='8':
            pairs = {"old":[('old','randoms20')],
                    'young':[('young','randoms20')]}
        if concept_set_type =='0':
            pairs = {"old":[('old','young')],
                    'young':[('old','young')],"men":[('men','women')], 'women':[('women','men')]} #young old shoudl affect, men women should not
        if concept_set_type=='1':
            pairs = {"old":[('old','randoms20')],
                    'young':[('young','randoms20')],"men":[('men','women')], 'women':[('women','men')]}
        if concept_set_type =='2':
            pairs = {"old":[('old','men_women_all_ages')],
                    'young':[('young','men_women_all_ages')],"men":[('men','women')], 'women':[('women','men')]}
        if concept_set_type =='3':
            pairs = {"old_men":[('old_men','men_women_all_ages')], 'young_women':[('young_women','men_women_all_ages')]}
        if concept_set_type =='4':
            pairs = {"old_men":[('old_men','men_women_all_ages')],
                    'young_women':[('young_women','men_women_all_ages')],"men":[('men','women')], 'women':[('women','men')]}
        if concept_set_type=='5': #class-wise training : young women shoudl not affect class women (currently model learnt young women only)
            pairs = {"young_women":[("young_women","old_women")],"old_men":[("old_men","young_men")]}
        if concept_set_type=='6':
            pairs = {"young_women":[("young_women","old_women"),("young_women","women")],"old_men":[("old_men","young_men"),("old_men","men")]}
        # if concept_set_type=='7':
        #     pairs = {"young_women":[("young_women","old_women")],"old_women":[("old_women","all_women")],"old_men":[("old_men","young_men")],"young_men":[("old_men","young_men")]}


    elif model_type =='cat_dog':
        if concept_set_type =='0':
            pairs = {'colors0': [('colors0', 'gray_mixed')],
                'colors1': [('colors1', 'gray_mixed')],
                'colors2': [('colors2', 'gray_mixed')],
                'colors3': [('colors3', 'gray_mixed')]}
        
        if concept_set_type =='1':
            pairs = {'bright_cats': [('bright_cats', 'mixed')],
                'bright_dogs': [('bright_dogs', 'mixed')],
                'dark_cats': [('dark_cats', 'mixed')],
                'dark_dogs': [('dark_dogs', 'mixed')]}

        if concept_set_type == '2':
           pairs = {'lightest_colorpatches0': [('lightest_colorpatches0', 'colors0')],
                'lightest_colorpatches1': [('lightest_colorpatches1', 'colors1')],
                'lightest_colorpatches2': [('lightest_colorpatches2', 'colors2')],
                'lightest_colorpatches3': [('lightest_colorpatches3', 'colors3')]}

    elif model_type =='clever_hans7':
        if concept_set_type == None or concept_set_type=='0':
            pairs = {'gray_1':[('gray_1','randoms')], 'gray_2':[('gray_2','randoms')], 'gray_3':[('gray_3','randoms')], 'gray_4':[('gray_4','randoms')], 'gray_5':[('gray_5','randoms')], 'metal':[('metal','randoms2')],'small_cyan_cubes':[('small_cyan_cubes', 'randoms')]}
        if concept_set_type == '2':
            pairs = {'gray_1':[('gray_1','randoms')], 'gray_2':[('gray_2','randoms')], 'gray_3':[('gray_3','randoms')], 'gray_4':[('gray_4','randoms')], 'gray_5':[('gray_5','randoms')]}
            with open('small_cubes.json','r') as f:
                pairs.update(json.load(f))
            with open('small_cyan_cubes.json','r') as f:
                pairs.update(json.load(f))
            pairs.update({'metal': [('metal', 'randoms20'),
                        ('metal', 'randoms21'),
                        ('metal', 'randoms22'),
                        ('metal', 'randoms23'),
                        ('metal', 'randoms24'),
                        ('metal', 'randoms25'),
                        ('metal', 'randoms26'),
                        ('metal', 'randoms27')]})


    elif model_type =='clever_hans':
        if concept_set_type == None or concept_set_type=='0':
            pairs = {'gray_1':[('gray_1','randoms')], 'gray_2':[('gray_2','randoms')], 'gray_3':[('gray_3','randoms')], 'gray_4':[('gray_4','randoms')], 'gray_5':[('gray_5','randoms')], 'metal':[('metal','randoms2')]}
        if concept_set_type == '2':
            pairs = {'gray_1':[('gray_1','randoms')], 'gray_2':[('gray_2','randoms')], 'gray_3':[('gray_3','randoms')], 'gray_4':[('gray_4','randoms')], 'gray_5':[('gray_5','randoms')]}
            pairs.update({'metal': [('metal', 'randoms20'),
                        ('metal', 'randoms21'),
                        ('metal', 'randoms22'),
                        ('metal', 'randoms23'),
                        ('metal', 'randoms24'),
                        ('metal', 'randoms25'),
                        ('metal', 'randoms26'),
                        ('metal', 'randoms27')]})

    
    elif model_type =='toy' or model_type =='toy_conv':
        pairs = {'star_pos_changing':[('star_pos_changing','randoms')], 'moon_pos_changing':[('moon_pos_changing','randoms')]}
    
    elif model_type=='cg' or model_type=='iiww':
        with open('pairs/pairs_large.pickle', 'rb') as handle:
            pairs = pickle.load(handle)
    
    elif model_type == 'decoymnist':
        if concept_set_type == '11':
            pairs = {'0_decoy_align': [('0_decoy_align', '0_decoy_conflict')],
                    '1_decoy_align': [('1_decoy_align', '1_decoy_conflict')],
                    '2_decoy_align': [('2_decoy_align', '2_decoy_conflict')],
                    '3_decoy_align': [('3_decoy_align', '3_decoy_conflict')],
                    '4_decoy_align': [('4_decoy_align', '4_decoy_conflict')],
                    '5_decoy_align': [('5_decoy_align', '5_decoy_conflict')],
                    '6_decoy_align': [('6_decoy_align', '6_decoy_conflict')],
                    '7_decoy_align': [('7_decoy_align', '7_decoy_conflict')],
                    '8_decoy_align': [('8_decoy_align', '8_decoy_conflict')],
                    '9_decoy_align': [('9_decoy_align', '9_decoy_conflict')],
                    'gray_digits':[('gray_digits','randoms')], 'gray_1':[('gray_1','randoms')], 'gray_2':[('gray_2','randoms')], 'gray_3':[('gray_3','randoms')], 'gray_4':[('gray_4','randoms')], 'gray_5':[('gray_5','randoms')]}

        if concept_set_type==None or concept_set_type=='-2':
            pairs = {'gray_digits':[('gray_digits','randoms')], 'gray_1':[('gray_1','randoms')], 'gray_2':[('gray_2','randoms')], 'gray_3':[('gray_3','randoms')], 'gray_4':[('gray_4','randoms')], 'gray_5':[('gray_5','randoms')]}
        if concept_set_type=='5':
            pairs = {'gray_1':[('gray_1','randoms')], 'gray_2':[('gray_2','randoms')], 'gray_3':[('gray_3','randoms')], 'gray_4':[('gray_4','randoms')], 'gray_5':[('gray_5','randoms')]}
        if concept_set_type=='10':
            pairs = {'0_decoy_align': [('0_decoy_align', '0_decoy_conflict')],
                    '1_decoy_align': [('1_decoy_align', '1_decoy_conflict')],
                    '2_decoy_align': [('2_decoy_align', '2_decoy_conflict')],
                    '3_decoy_align': [('3_decoy_align', '3_decoy_conflict')],
                    '4_decoy_align': [('4_decoy_align', '4_decoy_conflict')],
                    '5_decoy_align': [('5_decoy_align', '5_decoy_conflict')],
                    '6_decoy_align': [('6_decoy_align', '6_decoy_conflict')],
                    '7_decoy_align': [('7_decoy_align', '7_decoy_conflict')],
                    '8_decoy_align': [('8_decoy_align', '8_decoy_conflict')],
                    '9_decoy_align': [('9_decoy_align', '9_decoy_conflict')]}
        return pairs
            
    elif model_type=='colormnist':
        print("********************************",concept_set_type)
        if concept_set_type=='8':
            pairs = {'biased_color0': [('biased_color0', 'randoms')],
            'biased_color1': [('biased_color1', 'randoms')],
            'biased_color2': [('biased_color2', 'randoms')],
            'biased_color3': [('biased_color3', 'randoms')],
            'biased_color4': [('biased_color4', 'randoms')],
            'biased_color5': [('biased_color5', 'randoms')],
            'biased_color6': [('biased_color6', 'randoms')],
            'biased_color7': [('biased_color7', 'randoms')],
            'biased_color8': [('biased_color8', 'randoms')],
            'biased_color9': [('biased_color9', 'randoms')]}
            return pairs
        if concept_set_type == '11':
            pairs = {'0': [('0', '0_conflict')],
                    '1': [('1', '1_conflict')],
                    '2': [('2', '2_conflict')],
                    '3': [('3', '3_conflict')],
                    '4': [('4', '4_conflict')],
                    '5': [('5', '5_conflict')],
                    '6': [('6', '6_conflict')],
                    '7': [('7', '7_conflict')],
                    '8': [('8', '8_conflict')],
                    '9': [('9', '9_conflict')],
                    'biased_color0': [('biased_color0', 'randoms')],
                    'biased_color1': [('biased_color1', 'randoms')],
                    'biased_color2': [('biased_color2', 'randoms')],
                    'biased_color3': [('biased_color3', 'randoms')],
                    'biased_color4': [('biased_color4', 'randoms')],
                    'biased_color5': [('biased_color5', 'randoms')],
                    'biased_color6': [('biased_color6', 'randoms')],
                    'biased_color7': [('biased_color7', 'randoms')],
                    'biased_color8': [('biased_color8', 'randoms')],
                    'biased_color9': [('biased_color9', 'randoms')]}
            return pairs
        if concept_set_type == '10':
            pairs = {'0': [('0', '0_conflict')],
                    '1': [('1', '1_conflict')],
                    '2': [('2', '2_conflict')],
                    '3': [('3', '3_conflict')],
                    '4': [('4', '4_conflict')],
                    '5': [('5', '5_conflict')],
                    '6': [('6', '6_conflict')],
                    '7': [('7', '7_conflict')],
                    '8': [('8', '8_conflict')],
                    '9': [('9', '9_conflict')]}
            return pairs
                    
        if concept_set_type == None or concept_set_type =='-2':
            pairs = {'color_digits':[('color_digits','randoms'),('color_digits','gray_digits')],'colors1':[('colors1','gray_1')],'colors2':[('colors2','gray_2')], 'colors3':[('colors3','gray_3')], 'colors4':[('colors4','gray_4')], 'colors5':[('colors5','gray_5')]}
        if concept_set_type == '2':
            pairs = {'colors1':[('colors1','gray_1')],'colors2':[('colors2','gray_2')], 'colors3':[('colors3','gray_3')], 'colors4':[('colors4','gray_4')], 'colors5':[('colors5','gray_5')]}
        if concept_set_type == '0': #red zero with 9 colored zeros
            pairs = {'0': [('0', '0_9colored')],
                    '1': [('1', '1_9colored')],
                    '2': [('2', '2_9colored')],
                    '3': [('3', '3_9colored')],
                    '4': [('4', '4_9colored')],
                    '5': [('5', '5_9colored')],
                    '6': [('6', '6_9colored')],
                    '7': [('7', '7_9colored')],
                    '8': [('8', '8_9colored')],
                    '9': [('9', '9_9colored')]}
            return pairs

        if concept_set_type == '1':
            pairs = {'0': [('0', '0_gray')],
                    '1': [('1', '1_gray')],
                    '2': [('2', '2_gray')],
                    '3': [('3', '3_gray')],
                    '4': [('4', '4_gray')],
                    '5': [('5', '5_gray')],
                    '6': [('6', '6_gray')],
                    '7': [('7', '7_gray')],
                    '8': [('8', '8_gray')],
                    '9': [('9', '9_gray')]}
            return pairs

        if concept_set_type=='7':
            pairs = {'biased_color0': [('biased_color0', 'randoms_colored')],
            'biased_color1': [('biased_color1', 'randoms_colored')],
            'biased_color2': [('biased_color2', 'randoms_colored')],
            'biased_color3': [('biased_color3', 'randoms_colored')],
            'biased_color4': [('biased_color4', 'randoms_colored')],
            'biased_color5': [('biased_color5', 'randoms_colored')],
            'biased_color6': [('biased_color6', 'randoms_colored')],
            'biased_color7': [('biased_color7', 'randoms_colored')],
            'biased_color8': [('biased_color8', 'randoms_colored')],
            'biased_color9': [('biased_color9', 'randoms_colored')]}
            return pairs

        
        
    elif model_type=='texturemnist':
        if concept_set_type=='11':
            pairs= {
            '0_texture_align': [('0_texture_align', '0_texture_conflict')],
            '1_texture_align': [('1_texture_align', '1_texture_conflict')],
            '2_texture_align': [('2_texture_align', '2_texture_conflict')],
            '3_texture_align': [('3_texture_align', '3_texture_conflict')],
            '4_texture_align': [('4_texture_align', '4_texture_conflict')],
            '5_texture_align': [('5_texture_align', '5_texture_conflict')],
            '6_texture_align': [('6_texture_align', '6_texture_conflict')],
            '7_texture_align': [('7_texture_align', '7_texture_conflict')],
            '8_texture_align': [('8_texture_align', '8_texture_conflict')],
            '9_texture_align': [('9_texture_align', '9_texture_conflict')],
            'texture_0': [('texture_0', 'randoms')],
            'texture_1': [('texture_1', 'randoms')],
            'texture_2': [('texture_2', 'randoms')],
            'texture_3': [('texture_3', 'randoms')],
            'texture_4': [('texture_4', 'randoms')],
            'texture_5': [('texture_5', 'randoms')],
            'texture_6': [('texture_6', 'randoms')],
            'texture_7': [('texture_7', 'randoms')],
            'texture_8': [('texture_8', 'randoms')],
            'texture_9': [('texture_9', 'randoms')]}

        if concept_set_type=='9':
            pairs= {
            'texture_0': [('texture_0', 'randoms')],
            'texture_1': [('texture_1', 'randoms')],
            'texture_2': [('texture_2', 'randoms')],
            'texture_3': [('texture_3', 'randoms')],
            'texture_4': [('texture_4', 'randoms')],
            'texture_5': [('texture_5', 'randoms')],
            'texture_6': [('texture_6', 'randoms')],
            'texture_7': [('texture_7', 'randoms')],
            'texture_8': [('texture_8', 'randoms')],
            'texture_9': [('texture_9', 'randoms')]}
    
    # elif model_type =='isic' or model_type =='isic_vgg':
    #     pairs = {'colors1':[('colors1','randoms2')],'colors2':[('colors2','randoms2')], 'colors3':[('colors3','randoms2')], 'colors4':[('colors4','randoms2')], 'colors5':[('colors5','randoms2')]}

    elif model_type =='isic' or model_type =='isic_vgg':
        pairs = {'colors1':[('colors1','gray_1')],'colors2':[('colors2','gray_2')], 'colors3':[('colors3','gray_3')], 'colors4':[('colors4','gray_4')], 'colors5':[('colors5','gray_5')]}
        # pairs = {'colors1':[('colors1','randoms2')],'colors2':[('colors2','randoms2')], 'colors3':[('colors3','randoms2')], 'colors4':[('colors4','randoms2')], 'colors5':[('colors5','randoms2')]}

    elif model_type =='cat_dog':
        with open('light_concept.json','r') as f:
            pairs = json.load(f)
        with open('dark_concept.json','r') as f:
            pairs.update(json.load(f))

    elif model_type =='imdb':
        with open('age_concept.json','r') as f:
            pairs = json.load(f)
    return pairs


def get_concepts(model_type, test_env_idx): 
    
    if model_type=='PACS':
        domains = {"A":"paintings", "C":"cartoons", "P":"photos", "S":"sketches"}
        domain_vals = list(domains.values())
        return domain_vals[:test_env_idx]+domain_vals[test_env_idx+1:]
    


class ImgsDataset2(Dataset):
    def __init__(self,model_type,shape, num_imgs=50,concept_set_type=None,test_env=0):
        self.model_type = model_type
        self.shape = shape
        # pairs = get_pairs(model_type,concept_set_type)
        # print(pairs)
        # concepts = set()
        # for p in pairs:
        #     concepts.add(p)
        #     for l1,l2 in pairs[p]:
        #         concepts.add(l1)
        #         concepts.add(l2)
        # self.concepts = list(concepts)

        self.concepts = get_concepts(model_type, test_env)
        self.num_imgs = num_imgs

    def __len__(self):
        return len(self.concepts)

    def __getitem__(self, idx):
        con = self.concepts[idx]
        path = DATA_PATH+'imgs_np_'+str(self.shape[1])+'by'+str(self.shape[2])+'/'
        # if not os.path.exists(path+con+'.npy'):
        # dump_imgs(tuple(self.shape[1:]),con,self.num_imgs)
        
        imgs = np.load(path+con+'.npy')
        # if self.model_type =='isic' or self.model_type=='isic_vgg':
        #     path = DATA_PATH+'imgs_np_450by600/'
        #     imgs = np.load(path+con+'.npy')
        # elif self.model_type =='clever_hans7':
        #     path = DATA_PATH+'imgs_np_224by224/'
        #     imgs = np.load(path+con+'.npy')

        # elif self.model_type =='clever_hans':
        #     path = DATA_PATH+'imgs_np_224by224/'
        #     imgs = np.load(path+con+'.npy')

        # elif self.model_type=='toy' or self.model_type=='toy_conv':
        #     save_path = DATA_PATH+'imgs_np_'+str(self.shape[0])+'by'+str(self.shape[1])+'_gray/'
        #     imgs = np.load(save_path+con+'.npy') #load all imgs and save student out for them
        #     imgs = (imgs[:,:,:,0] + imgs[:,:,:,1] + imgs[:,:,:,2])/3 #convert rgb to gray: https://www.baeldung.com/cs/convert-rgb-to-grayscale
        #     imgs = np.expand_dims(imgs, axis=3)
            
        # elif self.shape == (28,28) or self.model_type =='colormnist' or self.model_type=='decoymnist' or self.model_type=='texturemnist':
        #     save_path = DATA_PATH+'imgs_np_28by28'+'/'
        #     if os.path.exists(save_path+con+'.npy'):
        #         imgs = np.load(save_path+con+'.npy')
        #     else:
        #         dump_imgs(self.shape, con)
        #         imgs = np.load(save_path+con+'.npy')
        
        # elif self.model_type=='cat_dog' or self.model_type=='faces' or self.model_type=='pacs':
        #     imgs = np.load(DATA_PATH+'imgs_np_224by224/'+con+'.npy')

        # if self.model_type=='decoymnist': #have grayscale imgs
        #     imgs = (imgs[:,:,:,0] + imgs[:,:,:,1] + imgs[:,:,:,2])/3 #convert rgb to gray: https://www.baeldung.com/cs/convert-rgb-to-grayscale
        #     imgs = np.expand_dims(imgs, axis=3)
        #     print(imgs.shape)
        print(imgs.shape)
        breakpoint()
        return imgs, con


class EncoMapISIC2NN(torch.nn.Module):
    '''
    Mainly for mapping to match dims
    '''
    def __init__(self):
        super(EncoMapISIC2NN, self).__init__()
        self.fc1 = nn.Linear(268800,4096)
        # self.fc2 = nn.Linear(67200, 16800)
        # self.fc3 = nn.Linear(16800,4096)
       
    def forward(self,x):
        x = self.fc1(x)
        # x = self.fc2(x)
        # x = self.fc3(x)
        return x

class DecoMapISIC2NN(torch.nn.Module):
    '''
    Mainly for mapping to match dims
    '''
    def __init__(self):
        super(DecoMapISIC2NN, self).__init__()
        self.fc1 = nn.Linear(4096, 268800)
        # self.fc2 = nn.Linear(67200, 16800)
        # self.fc3 = nn.Linear(268800,67200)

       
    def forward(self,x):
        x = self.fc1(x)
        # x = self.fc2(x)
        # x = self.fc3(x)
        return x

class EncoMapToyConvNN(torch.nn.Module):
    '''
    Mainly for mapping to match dims
    '''
    def __init__(self):
        super(EncoMapToyConvNN, self).__init__()
        self.conv = nn.Conv2d(32,64,(32,32),(7,7))
       
    def forward(self,x):
        x = self.conv(x)
        return x

class EncoMapCGNN(torch.nn.Module):
    '''
    Mainly for mapping to match dims
    '''
    def __init__(self):
        super(EncoMapCGNN, self).__init__()
        self.conv = nn.Conv2d(1,64,(71,55),(7,7))
       
    def forward(self,x):
        x = self.conv(x)
        return x

    
class DecoMapCGNN(torch.nn.Module):
    '''
    Mainly for mapping to match dims
    '''
    def __init__(self):
        super(DecoMapCGNN, self).__init__()
        self.conv = nn.ConvTranspose2d(64,1,(71,55),(7,7))
       
    def forward(self,x):
        x = self.conv(x)
        return x

class DecoMapToyConvNN(torch.nn.Module):
    '''
    Mainly for mapping to match dims
    '''
    def __init__(self):
        super(DecoMapToyConvNN, self).__init__()
        self.conv = nn.ConvTranspose2d(64,32,(32,32),(7,7))
       
    def forward(self,x):
        x = self.conv(x)
        return x

class EncoMapISICNN(torch.nn.Module):
    '''
    Mainly for mapping to match dims
    '''
    def __init__(self):
        super(EncoMapISICNN, self).__init__()
        self.conv = nn.Conv2d(64,512,(3,2),(2,2))
       
    def forward(self,x):
        x = self.conv(x)
        return x


class EncoMapCDNN(torch.nn.Module):
    '''
    Mainly for mapping to match dims
    '''
    def __init__(self):
        super(EncoMapCDNN, self).__init__()
        self.conv = nn.Conv2d(64,512,(10,10),(3,3))
       
    def forward(self,x):
        x = self.conv(x)
        return x

class DecoMapCDNN(torch.nn.Module):
    '''
    Mainly for mapping to match dims
    '''
    def __init__(self):
        super(DecoMapCDNN, self).__init__()
        self.conv = nn.ConvTranspose2d(512,64,(10,10),(3,3))
       
    def forward(self,x):
        x = self.conv(x)
        return x

class EncoMapResnet50NN(torch.nn.Module):
    '''
    Mainly for mapping to match dims
    '''
    def __init__(self):
        super(EncoMapResnet50NN, self).__init__()
        self.conv = nn.Conv2d(64,2048,(10,10),(3,3))
       
    def forward(self,x):
        x = self.conv(x)
        return x

class DecoMapResnet50NN(torch.nn.Module):
    '''
    Mainly for mapping to match dims
    '''
    def __init__(self):
        super(DecoMapResnet50NN, self).__init__()
        self.conv = nn.ConvTranspose2d(2048,64,(10,10),(3,3))
       
    def forward(self,x):
        x = self.conv(x)
        return x

class DecoMapISICNN(torch.nn.Module):
    '''
    Mainly for mapping to match dims
    '''
    def __init__(self):
        super(DecoMapISICNN, self).__init__()
        self.conv = nn.ConvTranspose2d(512,64,(3,2),(2,2))
       
    def forward(self,x):
        x = self.conv(x)
        return x

class EncoMapCHNN(torch.nn.Module):
    '''
    Mainly for mapping to match dims
    '''
    def __init__(self):
        super(EncoMapCHNN, self).__init__()
        self.conv = nn.Conv2d(64,256,(2,2),(2,2))
       
    def forward(self,x):
        x = self.conv(x)
        return x

class DecoMapCHNN(torch.nn.Module):
    '''
    Mainly for mapping to match dims
    '''
    def __init__(self):
        super(DecoMapCHNN, self).__init__()
        self.conv = nn.ConvTranspose2d(256,64,(2,2),(2,2))
       
    def forward(self,x):
        x = self.conv(x)
        return x


class EncoMapMNISTNN(torch.nn.Module):
    '''
    Mainly for mapping to match dims
    '''
    def __init__(self, in_ch, out_ch, kernel, stride):
        # in_ch=50, out_ch = 64, kernel=(6,6), stride=(1,1)
        super(EncoMapMNISTNN, self).__init__()
        self.conv = nn.Conv2d(in_ch,out_ch,kernel,stride)
       
    def forward(self,x):
        x = self.conv(x)
        return x

class DecoMapMNISTNN(torch.nn.Module):
    '''
    Mainly for mapping to match dims
    '''
    def __init__(self,in_ch = 64, out_ch = 50, kernel=(6,6), stride=(1,1)):
        super(DecoMapMNISTNN, self).__init__()
        self.conv = nn.ConvTranspose2d(in_ch,out_ch,kernel,stride)
       
    def forward(self,x):
        x = self.conv(x)
        return x

class EncoMapMNISTConvNN(torch.nn.Module):
    '''
    Mainly for mapping to match dims
    '''
    def __init__(self):
        super(EncoMapMNISTConvNN, self).__init__()
        self.conv = nn.Conv2d(20,64,(20,20),(2,2))
       
    def forward(self,x):
        x = self.conv(x)
        return x

def get_map_mod_enco(model_type,dset):
  if model_type=='colormnist' and dset=='mnistconv1':
    map_to_stu_nn = DecoMapMNISTConvNN()
  elif model_type=='colormnist' or model_type=='texturemnist':
    map_to_stu_nn = DecoMapMNISTNN()
  elif model_type =='decoymnist':
    map_to_stu_nn = DecoMapDMNISTNN()
  elif model_type == 'toy':
    map_to_stu_nn = DecoMapToyNN()
  elif model_type == 'clever_hans' or model_type == 'clever_hans7':
    map_to_stu_nn = EncoMapCHNN()
  elif model_type =='isic_vgg':
    map_to_stu_nn = EncoMapISICNN()
  elif model_type=='cat_dog' or model_type=='faces':
    map_to_stu_nn = EncoMapCDNN()
  elif model_type=='pacs':
    map_to_stu_nn = EncoMapResnet50NN()
  return map_to_stu_nn



class DecoMapMNISTConvNN(torch.nn.Module):
    '''
    Mainly for mapping to match dims
    '''
    def __init__(self):
        super(DecoMapMNISTConvNN, self).__init__()
        self.conv = nn.ConvTranspose2d(64,20,(20,20),(2,2))
       
    def forward(self,x):
        x = self.conv(x)
        return x

class EncoMapToyNN(torch.nn.Module):
    '''
    Mainly for mapping to match dims
    '''
    def __init__(self):
        super(EncoMapToyNN, self).__init__()
        self.fc = nn.Linear(1024, 40000)
       
    def forward(self,x):
        x = self.fc(x)
        return x

class DecoMapToyNN(torch.nn.Module):
    '''
    Mainly for mapping to match dims
    '''
    def __init__(self):
        super(DecoMapToyNN, self).__init__()
        self.fc = nn.Linear(40000, 1024)
       
    def forward(self,x):
        x = self.fc(x)
        return x

class EncoMapDMNISTNN(torch.nn.Module):
    '''
    Mainly for mapping to match dims
    '''
    def __init__(self):
        super(EncoMapDMNISTNN, self).__init__()
        self.conv = nn.Conv2d(50,64,(4,4),stride=2)
       
    def forward(self,x):
        x = self.conv(x)
        return x

class DecoMapDMNISTNN(torch.nn.Module):
    '''
    Mainly for mapping to match dims
    '''
    def __init__(self):
        super(DecoMapDMNISTNN, self).__init__()
        self.conv = nn.ConvTranspose2d(64,50,(4,4),stride=2)
       
    def forward(self,x):
        x = self.conv(x)
        return x


class NN(torch.nn.Module):
    def __init__(self,input_size=512, num_classes=2):
        super(NN,self).__init__()
        self.linear = torch.nn.Linear(input_size,num_classes)
        self.relu = nn.ReLU()
    
    def forward(self,feature):
        # feature = self.relu(self.conv1(feature))
        # feature = feature.view(-1,512)
        # print("fea",feature.shape)
        output = torch.sigmoid(self.linear(feature))
        return output

class CAV_NN(object):
  """CAV class contains methods for concept activation vector (CAV).

  CAV represents semenatically meaningful vector directions in
  network's embeddings (bottlenecks).
  """
  @staticmethod
  def cav_key(concepts, bottleneck, model_type, alpha):
    """A key of this cav (useful for saving files).

    Args:
      concepts: set of concepts used for CAV
      bottleneck: the bottleneck used for CAV
      model_type: the name of model for CAV
      alpha: a parameter used to learn CAV

    Returns:
      a string cav_key
    """
    return '-'.join([str(c) for c in concepts
                    ]) + '-' + bottleneck + '-' + model_type + '-' + str(alpha)

 
  @staticmethod
  def _create_cav_training_set_(concepts, bottleneck, acts):
    """Flattens acts, make mock-labels and returns the info.

    Labels are assigned in the order that concepts exists.

    Args:
        concepts: names of concepts
        bottleneck: the name of bottleneck where acts come from
        acts: a dictionary that contains activations
    Returns:
        x -  flattened acts
        labels - corresponding labels (integer)
        labels2text -  map between labels and text.
    """
    x = []
    labels = []
    labels2text = {}
    # to make sure postiive and negative examples are balanced,
    # truncate all examples to the size of the smallest concept.
    pt = []
    for concept in concepts:
      try:
        pt.append(acts[concept].shape[0])
      except:
        print(concept)
    min_data_points = torch.min(torch.tensor(pt))
    for i, concept in enumerate(concepts):
        ac = acts[concept][:min_data_points].reshape(min_data_points, -1)
        x.extend(ac)
        labels.extend([i] * min_data_points)
        labels2text[i] = concept

    labels = torch.tensor(labels).long()
    x = torch.stack(x)
    # print("x",x.shape, labels.shape)
    return x, labels, labels2text

  def __init__(self, concepts, bottleneck, save_path=None):
    """Initialize CAV class.

    Args:
      concepts: set of concepts used for CAV
      bottleneck: the bottleneck used for CAV
      hparams: a parameter used to learn CAV
      save_path: where to save this CAV
    """
    self.concepts = concepts
    self.bottleneck = bottleneck
    self.save_path = save_path

  def train(self, acts, reg_type):
    """Train the CAVs from the activations.

    Args:
      acts: is a dictionary of activations. In particular, acts takes for of
            {'concept1':{'bottleneck name1':[...act array...],
                         'bottleneck name2':[...act array...],...
             'concept2':{'bottleneck name1':[...act array...],
    Raises:
      ValueError: if the model_type in hparam is not compatible.
    """
    x, labels, labels2text = CAV_NN._create_cav_training_set_(
        self.concepts, self.bottleneck, acts)
    # print("here",x.shape, labels.shape)
    # if reg_type=='linear':
    #   model = LinearRegression(input_size=x.shape[1],num_classes=2)
    # if reg_type=='linear':
    
    
    self.accuracies,  st_dict, cavs = self._train_model(x, labels, labels2text, reg_type)
    print("concept ",self.concepts, " acuracies ",self.accuracies)
    self.cavs = cavs
    # breakpoint()
    return st_dict

  
  def get_direction(self, concept):
    """Get CAV direction.
    Args:
      concept: the conept of interest

    Returns:
      CAV vector.
    """
    return self.cavs[self.concepts.index(concept)]

  def _train_model(self, x, y, labels2text,reg_type):
    """Train a model to get CAVs.

    Modifies lm by calling the lm.fit functions. The cav coefficients are then
    in lm._coefs.

    Args:
      lm: An sklearn linear_model object. Can be linear regression or
        logistic regression. Must support .fit and ._coef.
      x: An array of training data of shape [num_data, data_dim]
      y: An array of integer labels of shape [num_data]
      labels2text: Dictionary of text for each label.

    Returns:
      Dictionary of accuracies of the CAVs.

    """
    # X_train, X_test, y_train, y_test = train_test_split(
    #     x, y, test_size=0.33, stratify=y, random_state=0)
    model = NN(num_classes=2,input_size=x.shape[1])
    model = model.cuda()

    train_size = int(0.7 * len(x))
    val_size = int(0.1 * len(x))
    val_end = val_size+train_size


    X_train, y_train = x[:train_size], y[:train_size]
    X_val, y_val = x[train_size:val_end], y[train_size:val_end]
    X_test, y_test = x[val_end:], y[val_end:]

    loss = torch.nn.CrossEntropyLoss()
    # elif reg_type == 'linear':
    #   loss = torch.nn.MSELoss()
    # elif reg_type=='nonlinear':
    #   loss = torch.nn.CrossEntropyLoss()
    
    # max_acc = 0
    # learning_rate = 0.001
    # optimizer_ = torch.optim.SGD(model.parameters(),lr=learning_rate)
    optimizer_ = torch.optim.Adam(model.parameters(), lr=1e-4)

    
    input_size = X_train.shape[1]
    num_epochs = 5000
    run = 0
    count = 0
    model.train()
    last_loss = np.inf
    patience = 20
    run = 0
    trigger_times = 0
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_, 'min')

    for epoch in tqdm(range(num_epochs)):
        X = X_train.cuda()
        y = y_train.cuda()

        # Nullify gradients w.r.t. parameters
        optimizer_.zero_grad()
        #forward propagation
        output = model(X)
        #compute loss based on obtained value and actual label
        compute_loss = loss(output,y.long())
        # backward propagation
        # if epoch!=num_epochs-1:
        compute_loss.backward()
        # update the parameters
        optimizer_.step()
        run+=1

        
        with torch.no_grad():
          val_out = model(X_val.cuda())
          val_loss = loss(val_out,y_val.long().cuda())
        
        scheduler.step(val_loss)
        
        # if val_loss > last_loss:
        #     trigger_times += 1
        # #     # torch.save(model.state_dict(),DATA_PATH+''+dset+'_model'+bias+str(run)+'2.pt')
        # #     run += 1
        #     if trigger_times >= patience:
        #         print('Early stopping at epoch ',epoch)
        #         break
        # else:
        #   trigger_times = 0
        # last_loss = val_loss

    with torch.no_grad():
        y_pred=model(X_test.cuda())
        preds = torch.argmax(y_pred,axis=1)
        total = 0.
        correct = 0.
        total += y_test.size(0)
        correct += (preds == y_test.cuda()).sum().item()
        acc = correct/total


    return {'overall':acc}, model.linear.state_dict(), model.linear.weight.detach().cpu().numpy()


def get_or_train_cav_nn(concepts,
                     bottleneck,
                     acts,
                     reg_type='linear'):
  """Gets, creating and training if necessary, the specified CAV.

  Assumes the activations already exists.

  Args:
    concepts: set of concepts used for CAV
            Note: if there are two concepts, provide the positive concept
                  first, then negative concept (e.g., ['striped', 'random500_1']
    bottleneck: the bottleneck used for CAV
    acts: dictionary contains activations of concepts in each bottlenecks
          e.g., acts[concept]
    cav_dir: a directory to store the results.
    cav_hparams: a parameter used to learn CAV
    overwrite: if set to True overwrite any saved CAV files.

  Returns:
    returns a CAV instance
  """
  cav_instance = CAV_NN(concepts, bottleneck)
  st_dict =cav_instance.train({c: acts[c] for c in concepts}, reg_type=reg_type)
  return cav_instance, st_dict

def dump_imgs(shape, con, num_imgs,col_type='rgb'):
    save_path = DATA_PATH+'imgs_np_'+str(shape[0])+'by'+str(shape[1])+'/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    imgs_loaded = []
    path = DATA_PATH+'IID_data/concepts/'+con+'/'
    print("tot imgs", len(os.listdir(path)))
    imgs_con_full = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    num_imgs = int(num_imgs)

    for file in imgs_con_full:
        im = load_image_from_file(path+file, shape=shape)
        if len(im)!=0:
            imgs_loaded.append(im)
        else:
            print("could not load",path+file)

    np.save(save_path+con, np.array(imgs_loaded))
    print(f"dumped img con {con} to {save_path+con}")
    
def add_rand_to_pairs(pairs, num_random_cons, faces=False):
    new_pairs = {}
    # new_pairs.update(pairs)
    for p in pairs:
        for pp in pairs[p]:
            con, neg_con = pp
            for neg_ex_con in range(num_random_cons):
                if faces:
                    neg_ex_con = 'randoms2'+str(neg_ex_con)
                else:
                    neg_ex_con = 'randoms_'+str(neg_ex_con)

                if p in new_pairs:
                    new_pairs[p].append((con,neg_ex_con))
                else:
                    new_pairs[p] = [(con,neg_ex_con)]

    for neg_ex_con in range(num_random_cons):
        for neg_ex_con2 in range(num_random_cons):
            if neg_ex_con != neg_ex_con2:
                if faces:
                    new_pairs[p].append(('randoms2'+str(neg_ex_con),'randoms2'+str(neg_ex_con2)))
                else:
                    new_pairs[p].append(('randoms_'+str(neg_ex_con),'randoms_'+str(neg_ex_con2)))


    return new_pairs
    
import glob
def dump_random_imgs(num_random_cons, num_imgs):
    rand_imgs = glob.glob(DATA_PATH+'IID_data/concepts/randoms/*.png')
    num_random_cons = 10
    shape = (28,28)
    cur = 0
    for i in range(num_random_cons):
        con = 'randoms_'+str(i)
        save_path = DATA_PATH+'imgs_np_28by28'+'/'
        imgs_loaded = []
        if (cur+num_imgs)>len(rand_imgs):
            print("tot ran cons",i)
            break
        imgs = rand_imgs[cur:cur+num_imgs]
        cur += num_imgs
        for file in imgs:
            im = load_image_from_file(file, shape=shape)
            imgs_loaded.append(im)
        np.save(save_path+con, np.array(imgs_loaded))


class ImgsDataset(Dataset):
    def __init__(self,model_type,shape, concept_set_type=None, include_randoms=True,num_imgs=44,pairs=None, transform=None):
        self.model_type = model_type
        self.shape = shape
        if pairs==None:
            pairs = get_pairs(model_type,concept_set_type)
        concepts = set()
        for p in pairs:
            concepts.add(p)
            for l1,l2 in pairs[p]:
                concepts.add(l1)
                if include_randoms:
                    concepts.add(l2)
                
        self.concepts = list(concepts)
        self.num_imgs = num_imgs
        self.transform = transform
    
    def __len__(self):
        return len(self.concepts)

    def __getitem__(self, idx):
        con = self.concepts[idx]
        path = DATA_PATH+f'imgs_np_{self.shape[0]}by{self.shape[0]}/'
        if os.path.exists(path+con+'.npy'):
            imgs = np.load(path+con+'.npy')
        else:
            dump_imgs(self.shape, con, self.num_imgs)
            imgs = np.load(path+con+'.npy')
       
        if self.model_type=='decoymnist': #have grayscale imgs
            try:
                imgs = (imgs[:,:,:,0] + imgs[:,:,:,1] + imgs[:,:,:,2])/3 #convert rgb to gray: https://www.baeldung.com/cs/convert-rgb-to-grayscale
                imgs = np.expand_dims(imgs, axis=3)
            except:
                imgs = np.expand_dims(imgs, axis=3)
            # imgs = imgs[:,:,:,0]
        
        # imgs = torch.from_numpy(imgs).float()
        imgs_ = []
        if self.transform:
            for im in imgs:
                im = self.transform(im)
                imgs_.append(im)
            imgs_ = torch.stack(imgs_)
            return imgs_, con
        else:
            return imgs, con



### mapping module utils
class EncoMapISIC2NN(torch.nn.Module):
    '''
    Mainly for mapping to match dims
    '''
    def __init__(self):
        super(EncoMapISIC2NN, self).__init__()
        self.fc1 = nn.Linear(268800,4096)
        # self.fc2 = nn.Linear(67200, 16800)
        # self.fc3 = nn.Linear(16800,4096)
       
    def forward(self,x):
        x = self.fc1(x)
        # x = self.fc2(x)
        # x = self.fc3(x)
        return x

class DecoMapISIC2NN(torch.nn.Module):
    '''
    Mainly for mapping to match dims
    '''
    def __init__(self):
        super(DecoMapISIC2NN, self).__init__()
        self.fc1 = nn.Linear(4096, 268800)
        # self.fc2 = nn.Linear(67200, 16800)
        # self.fc3 = nn.Linear(268800,67200)

       
    def forward(self,x):
        x = self.fc1(x)
        # x = self.fc2(x)
        # x = self.fc3(x)
        return x

class EncoMapToyConvNN(torch.nn.Module):
    '''
    Mainly for mapping to match dims
    '''
    def __init__(self):
        super(EncoMapToyConvNN, self).__init__()
        self.conv = nn.Conv2d(32,64,(32,32),(7,7))
       
    def forward(self,x):
        x = self.conv(x)
        return x

class EncoMapCGNN(torch.nn.Module):
    '''
    Mainly for mapping to match dims
    '''
    def __init__(self):
        super(EncoMapCGNN, self).__init__()
        self.conv = nn.Conv2d(1,64,(71,55),(7,7))
       
    def forward(self,x):
        x = self.conv(x)
        return x

    
class DecoMapCGNN(torch.nn.Module):
    '''
    Mainly for mapping to match dims
    '''
    def __init__(self):
        super(DecoMapCGNN, self).__init__()
        self.conv = nn.ConvTranspose2d(64,1,(71,55),(7,7))
       
    def forward(self,x):
        x = self.conv(x)
        return x

class DecoMapToyConvNN(torch.nn.Module):
    '''
    Mainly for mapping to match dims
    '''
    def __init__(self):
        super(DecoMapToyConvNN, self).__init__()
        self.conv = nn.ConvTranspose2d(64,32,(32,32),(7,7))
       
    def forward(self,x):
        x = self.conv(x)
        return x

class EncoMapISICNN(torch.nn.Module):
    '''
    Mainly for mapping to match dims
    '''
    def __init__(self):
        super(EncoMapISICNN, self).__init__()
        self.conv = nn.Conv2d(64,512,(3,2),(2,2))
       
    def forward(self,x):
        x = self.conv(x)
        return x


class EncoMapCDNN(torch.nn.Module):
    '''
    Mainly for mapping to match dims
    '''
    def __init__(self):
        super(EncoMapCDNN, self).__init__()
        self.conv = nn.Conv2d(64,512,(10,10),(3,3))
       
    def forward(self,x):
        x = self.conv(x)
        return x

class DecoMapCDNN(torch.nn.Module):
    '''
    Mainly for mapping to match dims
    '''
    def __init__(self):
        super(DecoMapCDNN, self).__init__()
        self.conv = nn.ConvTranspose2d(512,64,(10,10),(3,3))
       
    def forward(self,x):
        x = self.conv(x)
        return x

class EncoMapResnet50NN(torch.nn.Module):
    '''
    Mainly for mapping to match dims
    '''
    def __init__(self):
        super(EncoMapResnet50NN, self).__init__()
        self.conv = nn.Conv2d(64,2048,(10,10),(3,3))
       
    def forward(self,x):
        x = self.conv(x)
        return x

class DecoMapResnet50NN(torch.nn.Module):
    '''
    Mainly for mapping to match dims
    '''
    def __init__(self):
        super(DecoMapResnet50NN, self).__init__()
        self.conv = nn.ConvTranspose2d(2048,64,(10,10),(3,3))
       
    def forward(self,x):
        x = self.conv(x)
        return x

class DecoMapISICNN(torch.nn.Module):
    '''
    Mainly for mapping to match dims
    '''
    def __init__(self):
        super(DecoMapISICNN, self).__init__()
        self.conv = nn.ConvTranspose2d(512,64,(3,2),(2,2))
       
    def forward(self,x):
        x = self.conv(x)
        return x

class EncoMapCHNN(torch.nn.Module):
    '''
    Mainly for mapping to match dims
    '''
    def __init__(self):
        super(EncoMapCHNN, self).__init__()
        self.conv = nn.Conv2d(64,256,(2,2),(2,2))
       
    def forward(self,x):
        x = self.conv(x)
        return x

class DecoMapCHNN(torch.nn.Module):
    '''
    Mainly for mapping to match dims
    '''
    def __init__(self):
        super(DecoMapCHNN, self).__init__()
        self.conv = nn.ConvTranspose2d(256,64,(2,2),(2,2))
       
    def forward(self,x):
        x = self.conv(x)
        return x

class EncoMapMNISTNN(torch.nn.Module):
    '''
    Mainly for mapping to match dims
    '''
    def __init__(self):
        super(EncoMapMNISTNN, self).__init__()
        self.conv = nn.Conv2d(50,64,(6,6),(1,1))
       
    def forward(self,x):
        x = self.conv(x)
        return x

class DecoMapMNISTNN(torch.nn.Module):
    '''
    Mainly for mapping to match dims
    '''
    def __init__(self):
        super(DecoMapMNISTNN, self).__init__()
        self.conv = nn.ConvTranspose2d(64,50,(6,6),(1,1))
       
    def forward(self,x):
        x = self.conv(x)
        return x

class EncoMapMNISTConvNN(torch.nn.Module):
    '''
    Mainly for mapping to match dims
    '''
    def __init__(self):
        super(EncoMapMNISTConvNN, self).__init__()
        self.conv = nn.Conv2d(20,64,(20,20),(2,2))
       
    def forward(self,x):
        x = self.conv(x)
        return x

class DecoMapMNISTConvNN(torch.nn.Module):
    '''
    Mainly for mapping to match dims
    '''
    def __init__(self):
        super(DecoMapMNISTConvNN, self).__init__()
        self.conv = nn.ConvTranspose2d(64,20,(20,20),(2,2))
       
    def forward(self,x):
        x = self.conv(x)
        return x

class EncoMapToyNN(torch.nn.Module):
    '''
    Mainly for mapping to match dims
    '''
    def __init__(self):
        super(EncoMapToyNN, self).__init__()
        self.fc = nn.Linear(1024, 40000)
       
    def forward(self,x):
        x = self.fc(x)
        return x

class DecoMapToyNN(torch.nn.Module):
    '''
    Mainly for mapping to match dims
    '''
    def __init__(self):
        super(DecoMapToyNN, self).__init__()
        self.fc = nn.Linear(40000, 1024)
       
    def forward(self,x):
        x = self.fc(x)
        return x

class EncoMapDMNISTNN(torch.nn.Module):
    '''
    Mainly for mapping to match dims
    '''
    def __init__(self):
        super(EncoMapDMNISTNN, self).__init__()
        self.conv = nn.Conv2d(50,64,(4,4),stride=2)
       
    def forward(self,x):
        x = self.conv(x)
        return x

class DecoMapDMNISTNN(torch.nn.Module):
    '''
    Mainly for mapping to match dims
    '''
    def __init__(self):
        super(DecoMapDMNISTNN, self).__init__()
        self.conv = nn.ConvTranspose2d(64,50,(4,4),stride=2)
       
    def forward(self,x):
        x = self.conv(x)
        return x

def get_mapping_module(model_type):
    if model_type =='pacs' or model_type=='resnet50':
        if model_type =='pacs':
            dset_type = 'pacs'
        if model_type =='resnet50':
            dset_type = 'textures'
        downconv_m = EncoMapResnet50NN()
        upconv_m = DecoMapResnet50NN()
        bottleneck_name = 'layer4.2.conv3'

    if model_type =='cg':
        dset_type = 'cg'
        downconv_m = DecoMapCGNN()
        upconv_m = EncoMapCGNN()
        bottleneck_name = 'model.upconv_model_1.4'

    if model_type =='toy_conv':
        dset_type ='toy_conv'
        downconv_m = DecoMapToyConvNN()
        upconv_m = EncoMapToyConvNN()
        bottleneck_name = 'c2'

    if model_type =='isic_vgg':
        dset_type ='isic_vgg'
        downconv_m = EncoMapISICNN()
        upconv_m = DecoMapISICNN()
        bottleneck_name = 'features.28'

    if model_type =='cat_dog':
        dset_type ='cat_dog'
        downconv_m = EncoMapCDNN()
        upconv_m = DecoMapCDNN()
        bottleneck_name = 'layer4.1.conv1'

    if model_type =='faces':
        dset_type ='faces'
        downconv_m = EncoMapCDNN()
        upconv_m = DecoMapCDNN()
        bottleneck_name = 'layer4.1.conv1'

    if model_type =='isic':
        dset_type = 'isic'
        bottleneck_name = '0'
        downconv_m = EncoMapISIC2NN()
        upconv_m = DecoMapISIC2NN()
        
    if model_type == 'clever_hans':
        dset_type = 'clever_hans'
        bottleneck_name = 'features.6.5.conv1'
        downconv_m = EncoMapCHNN()
        upconv_m = DecoMapCHNN()

    if model_type == 'clever_hans7':
        dset_type = 'clever_hans7'
        bottleneck_name = 'features.6.5.conv1'
        downconv_m = EncoMapCHNN()
        upconv_m = DecoMapCHNN()


    if model_type == 'toy':
        dset_type = 'toy'
        bottleneck_name = 'fc2'
        upconv_m = EncoMapToyNN()
        downconv_m = DecoMapToyNN()

    if model_type == 'colormnist' or model_type=='texturemnist':
        
        dset_type = 'mnist'
        if model_type=='texturemnist':
            dset_type = 'tmnist'

        bottleneck_name = 'conv2'
        if bottleneck_name=='conv2':
            upconv_m = EncoMapMNISTNN()
            downconv_m = DecoMapMNISTNN()
        elif bottleneck_name =='conv1':
            dset_type = dset_type+bottleneck_name
            upconv_m = EncoMapMNISTConvNN()
            downconv_m = DecoMapMNISTConvNN()
    
    elif model_type == 'decoymnist':
        dset_type = 'dmnist'
        bottleneck_name = 'conv2'
        upconv_m = EncoMapDMNISTNN()
        downconv_m = DecoMapDMNISTNN()
    
    # elif model_type =='pacs':
    #     dset_type = 'pacs'
    #     bottleneck_name = 'conv2'
    #     # pairs = {'colors1':('colors1','randoms'),'colors2':('colors2','randoms'),'colors3':('colors3','randoms'),'colors4':('colors4','randoms'),'colors5':('colors5','randoms')}
    #     upconv_m = EncoMapPACSNN()
    #     downconv_m = DecoMapPACSNN()
    return upconv_m, downconv_m, dset_type, bottleneck_name

# from utils.networks_classi import *
from os.path import join as oj

def get_model_save_name(opt):
    model_save_name = opt.model_type+opt.dset+"update_cav"+opt.update_cav
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
    return model_save_name

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def get_vanilla_model(model_type):
    if model_type == 'faces':
        shape = (224,224)
        model = resnet18(pretrained=False)
        model.fc = nn.Linear(512,2)
        model.load_state_dict(torch.load("/media/Data2/avani.gupta/new_checkpoints/acc59.0best_val_acc99.5highest_epoch0iter0facesfacesp8nimg150lr0.01rr0.3wtcav5bs44pwt0.3upr1cd0precalc1up1scratch0uknn1cwt0s42corr"))

    if model_type == 'decoymnist':
        model = MNISTDecoyNet()
        model.cuda()
        shape = (28,28)
        model.load_state_dict(torch.load('mnist/DecoyMNIST/orig_model_decoyMNIST_.pt'))

    if model_type == 'colormnist':
        model = MNISTColorNet()
        model.cuda()
        shape = (28,28)
        model.load_state_dict(torch.load('mnist/ColorMNIST/orig_model_colorMNIST.pt'))
    
    if model_type == 'texturemnist':
        model = MNISTColorNet()
        model.cuda()
        shape = (28,28)
        model.load_state_dict(torch.load('/home/avani/tcav_pt/main_files/methodvanilla_tex.pt'))

    return model, shape  

def get_model_and_data(model_type, opt, kwargs):
    X_full, y_full = None, None
    val_x, val_y =  None, None
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

    if model_type == 'decoymnist':
        dpath = 'dep/data/DecoyMNIST/'
        X_full = torch.Tensor(np.load(oj(dpath, "train_x_decoy.npy")))
        y_full = torch.Tensor(np.load(oj(dpath, "train_y.npy"))).type(torch.int64)
        complete_dataset = tutils.TensorDataset(X_full, y_full) # create your datset

        num_train = int(len(complete_dataset)*.9)
        num_test = len(complete_dataset)  - num_train 
        torch.manual_seed(0)
        train_dataset, val_dataset,= torch.utils.data.random_split(complete_dataset, [num_train, num_test])
        train_loader = tutils.DataLoader(train_dataset, batch_size=opt.num_imgs, shuffle=True, **kwargs) # create your dataloader
        val_loader = tutils.DataLoader(val_dataset, batch_size=opt.num_imgs, shuffle=True, **kwargs) # create your dataloader

        test_x_tensor = torch.Tensor(np.load(oj(dpath, "test_x_decoy.npy")))
        test_y_tensor = torch.Tensor(np.load(oj(dpath, "test_y.npy"))).type(torch.int64)
        test_dataset = tutils.TensorDataset(test_x_tensor,test_y_tensor) # create your datset
        test_loader = tutils.DataLoader(test_dataset, batch_size=opt.num_imgs, shuffle=True, **kwargs) # create your dataloader
        
        model = MNISTDecoyNet()
        model.cuda()
        shape = (28,28)
        if not opt.train_from_scratch:
            print("loading baseline model")
            # model.load_state_dict(torch.load(DATA_PATH+'best_checkpoints/acc98.98best47decoymnistdmnistpairs_vals5num_imgs150lr0.58regularizer_rate0.3wtcav0.93batch_size256per_proto_mean_wt0.35use_proto0use_cdepcolor1use_precalc_proto1update_proto1train_from_scratch1use_knn_proto0class_wise_training0final_run'))
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
        train_loader = tutils.DataLoader(train_dataset,batch_size=opt.batch_size,shuffle=True)
        val_loader = tutils.DataLoader(val_dataset,batch_size=opt.batch_size)
        test_loader = tutils.DataLoader(test_dataset,batch_size=opt.batch_size)
        model = MNISTColorNet()
        model.cuda()
        shape = (28,28)

        if not opt.train_from_scratch:
            # model.load_state_dict(torch.load(DATA_PATH+'new_checkpoints/acc46.64best_val_acc45.7highest_epoch0iter130colormnistcolormnistp8nimg150lr0.01rr0.3wtcav5bs44pwt0.3upr1cd0precalc1up1scratch0uknn1cwt0s42corr'))
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
        train_loader = tutils.DataLoader(train_dataset,batch_size=opt.batch_size,shuffle=True)
        val_loader = tutils.DataLoader(val_dataset,batch_size=opt.batch_size)
        test_loader = tutils.DataLoader(test_dataset,batch_size=opt.batch_size)
        model = MNISTColorNet()
        model.cuda()
        shape = (28,28)
        if not opt.train_from_scratch:
            model.load_state_dict(torch.load('/home/avani/tcav_pt/main_files/methodvanilla_tex.pt'))
            # model.load_state_dict(torch.load('acc49.71best_val_acc56.53highest_epoch0iter56texturemnisttmnistp8nimg100lr0.01rr0.3wtcav5bs44pwt0.3upr1cd1precalc1up1scratch1uknn1cwt0s42corr'))
        
    
    return model, bottleneck_name, train_loader, val_loader, test_loader, X_full, y_full, val_x, val_y, shape
        








