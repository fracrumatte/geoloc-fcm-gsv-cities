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
    raise Exception(f'Please make sure the path {DATASET_ROOT} to SanFrancisco dataset is correct')

if not path_obj.joinpath('ref') or not path_obj.joinpath('query'):
    raise Exception(f'Please make sure the directories query and ref are situated in the directory {DATASET_ROOT}')

class SF_Dataset(Dataset):
    def __init__(self, which_ds='sf_val', input_transform = None):
        
        assert which_ds.lower() in ['sf_val', 'sf_test']
        
        self.input_transform = input_transform

        # reference images names
        self.dbImages = np.load(GT_ROOT+f'SanFrancisco/{which_ds}_dbImages.npy', allow_pickle=True)  # I added allow_pickle=true 
        
        # query images names
        self.qImages = np.load(GT_ROOT+f'SanFrancisco/{which_ds}_qImages.npy', allow_pickle=True)   # I added allow_pickle=true 
        
        # ground truth
        self.ground_truth = np.load(GT_ROOT+f'SanFrancisco/{which_ds}_gt.npy', allow_pickle=True)
        
        # reference images then query images
        self.images = np.concatenate((self.dbImages, self.qImages))
        
        self.num_references = len(self.dbImages)
        self.num_queries = len(self.qImages)

    def getDatasetRootPath(self,title):
        ds_db_path = Path(DATASET_ROOT+'/database/'+title+'.jpg')
        ds_q_path = Path(DATASET_ROOT+'/queries/'+title+'.jpg')
        if ds_db_path.exists():
            return ds_db_path
        else:
            return ds_q_path   
    
    def __getitem__(self, index):

        ds_root_path= self.getDatasetRootPath(self.images[index][7])
        img = Image.open(ds_root_path)

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)