from pathlib import Path
import numpy as np
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset

BASE_PATH = '/content/drive/MyDrive/geoloc_fcm/extracted_datasets/sf_xs/'


class SF_Dataset(Dataset):
    df = pd.read_csv(BASE_PATH+'dataset_sf_val_completo.csv')
    pass