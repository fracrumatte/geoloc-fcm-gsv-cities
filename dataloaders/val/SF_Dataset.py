from pathlib import Path
import numpy as np
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset

BASE_PATH = '/content/drive/MyDrive/geoloc_fcm/extracted_datasets/sf_xs/'




DATASET_ROOT = '/content/drive/MyDrive/geoloc_fcm/extracted_datasets/sf_xs/val' #colab path 
# DATASET_ROOT = 'geoloc-fcm-gsv-cities/datasets/SanFrancisco'  #local path
GT_ROOT = '/content/drive/MyDrive/geoloc_fcm/geoloc-fcm-gsv-cities/datasets/' #colab path
# GT_ROOT = 'geoloc-fcm-gsv-cities/datasets/SanFrancisco'    #local path
# BECAREFUL, this is the ground truth that comes with GSV-Cities

path_obj = Path(DATASET_ROOT)
if not path_obj.exists():
    raise Exception(f'Please make sure the path {DATASET_ROOT} to Nordland dataset is correct')

if not path_obj.joinpath('ref') or not path_obj.joinpath('query'):
    raise Exception(f'Please make sure the directories query and ref are situated in the directory {DATASET_ROOT}')

class SF_Dataset(Dataset):
    def __init__(self, which_ds='sf_val', input_transform = None):
        
        assert which_ds.lower() in ['sf_val', 'sf_test']
        
        self.input_transform = input_transform

        # reference images names
        self.dbImages = np.load(GT_ROOT+f'SanFrancisco/{which_ds}_dbImages.npy')    
        
        # query images names
        self.qImages = np.load(GT_ROOT+f'SanFrancisco/{which_ds}_qImages.npy')
        
        # ground truth
        self.ground_truth = np.load(GT_ROOT+f'SanFrancisco/{which_ds}_gt.npy', allow_pickle=True)
        
        # reference images then query images
        self.images = np.concatenate((self.dbImages, self.qImages))
        
        self.num_references = len(self.dbImages)
        self.num_queries = len(self.qImages)
        
    
    def __getitem__(self, index):
        img = Image.open(DATASET_ROOT+self.images[index])

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)