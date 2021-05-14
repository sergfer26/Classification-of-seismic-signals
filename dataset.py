from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import torch
from os import listdir
from os.path import isfile, join
import numpy as np



CLASS = {'Exhalation': 0, 'Explosions': 1, 'Tremor': 2, 'VTs':3}

# key: first word of the file name before the under score, value: label 


get_label = lambda name: CLASS.get(name) 
convert_rgb = lambda image: image.convert('RGB')


class LabeledSpectrograms1(Dataset):
    def __init__(self, dir):
        self.dir = dir
        FILES = [f for f in listdir(dir) if isfile(join(dir, f))] # list of file names 
        self.X = list()
        self.Y = list()
        self.read_image = lambda file: Image.open(os.path.join(self.dir, file))
        for f in FILES:
            clase = f.split('_')[0] 
            self.X.append(f)
            self.Y.append(CLASS[clase])
            
    def __len__(self):
        return len(self.Y)
    
    def __getitem__(self, index):
        image = self.X[index]
        x = np.array(self.read_image(image).convert('RGB'), dtype=np.float64)
        x = torch.tensor(x.T)
        y = torch.tensor(float(self.Y[index]))
        return x, y


class LabeledSpectrograms3(Dataset):
    def __init__(self, dir):
        self.dir = dir
        FILES = [f for f in listdir(dir) if isfile(join(dir, f))] # list of file names 
        tags = list()
        self.read_image = lambda file: Image.open(os.path.join(self.dir, file))
        for f in FILES:
            try:
                clase, date, stn, ch, _, _ = f.split('_')
            except:
                clase, date, stn, ch, _ = f.split('_')
            tags.append(clase + '_' + date + '_' + stn)
        
        tags = set(tags)
        
        self.X = list()
        self.Y = list()
        
        for tag in tags:
            clase, _, _ = tag.split('_')
            other_files = [f for f in FILES if f.startswith(tag)]
            
            if len(other_files) == 3:
                self.X.append(other_files)
                self.Y.append(CLASS[clase])
            
    def __len__(self):
        return len(self.Y)
    
    def __getitem__(self, index):
        images = self.X[index]
        arrays = [np.array(self.read_image(im).convert('RGB'), dtype=np.float64).T for im in images]
        x = np.concatenate(arrays, axis=-3)
        x = torch.tensor(x)
        y = torch.tensor(float(self.Y[index]))
        return x, y