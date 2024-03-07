import numpy as np # linear algebra
import torch
from torch import nn, optim
from torchvision import datasets, models, transforms
from torch import functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image
import os
import glob
from tqdm import tqdm

class IMDBTestDataSet(Dataset):
    def __init__(self, transforms):
        
        self.data_dir = '/media/Data2/avani.gupta/imdb_crop/'
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
        self.data_dir = '/media/Data2/avani.gupta/dog_cats/test/'
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
            for n in glob.glob('/media/Data2/avani.gupta/'+dset+'_model_transfer_learn'+bias+'*.pt'):
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
