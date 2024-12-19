import os
import numpy as np
import csv 

import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import torch

from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image

####################################################
################## TRAIN/VAL DATASET ###############
####################################################

'''
Transforming image from RGB scale into grayscale.
[3,256,256] --> [1, 256, 256]
'''
import torch

class GrayscaleTransform:
    def __init__(self, mean=(0.485,), std=(0.229,)):
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

    def normalize(self, image):
        # Ensure mean and std are reshaped for broadcasting
        mean = self.mean.view(1, 1, 1)
        std = self.std.view(1, 1, 1)
        return (image - mean) / std

    def to_grayscale(self, rgb_image):
        if rgb_image.ndim != 3 or rgb_image.shape[0] != 3:
            raise ValueError("Expected RGB image of shape (3, H, W), but got shape {}".format(rgb_image.shape))
        
        # Convert RGB to grayscale using weighted sum
        weights = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32)
        grayscale_image = torch.einsum('chw,c->hw', rgb_image, weights)
        
        # Add channel dimension to match shape (1, H, W)
        return grayscale_image.unsqueeze(0)

    def __call__(self, rgb_image):
        print(f"Input image shape: {rgb_image.shape}")
        # grayscale_image = self.to_grayscale(rgb_image)
        return self.normalize(rgb_image)



'''
Dataset class responsible for loading train/validation dataset into the pipeline
'''
class ChessDataset(Dataset):
    def __init__(self, img_dir, input_dim, file_ids, tokenizer):
        self.img_dir = img_dir
        self.img_extention = ".png"
        self.label_extention = ".txt"
        self.list_of_files = file_ids
        self.transform = transforms.Compose([
                            transforms.ConvertImageDtype(torch.float),
                            transforms.CenterCrop(input_dim),
                            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Convert grayscale to 3-channel
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
                            ])
        self.tokenizer = tokenizer
        
    def __len__(self,):
        return len(self.list_of_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, (self.list_of_files[idx] + self.img_extention))
        label_path = os.path.join(self.img_dir,  (self.list_of_files[idx] + self.label_extention))

        #Img
        image = read_image(img_path)
        if self.transform:
            image = self.transform(image)
        
        if hasattr(self, 'transform_A'):
            image = image.permute(1, 2, 0).numpy() 
            augmented = self.transform_A(image=image)
            image = augmented["image"]
            # image = image.unsqueeze(0)
            # image = image.repeat(3, 1, 1)
            

        # Label
        f = open(label_path, "r")
        label = f.read()
        label = self.tokenizer.encode(label)
        label = torch.from_numpy(label)

        return image, label
    
### Support functions
    
def get_image_ids(path):
    file_names = os.listdir(path)
    files = [f[:-4] for f in file_names if f.endswith(".png")]
    return files

def split_array(arr, ratio, debug=False):
    if debug:
        arr = arr[:200]

    # Ensure the array is a numpy array for easy slicing
    arr = np.array(arr)
    
    # Calculate the split index
    split_idx = int(len(arr) * ratio)
    
    # Split the array
    part1 = arr[:split_idx]
    part2 = arr[split_idx:]
    
    return part1, part2

####################################################
################## TEST DATASET ####################
####################################################

import os
import numpy as np
import csv 

import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import torch

from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image

class TestDataset(Dataset):
    def __init__(self, validation_csv_path, img_dir_path, tokenizer, input_dim):
        self.validation_csv_path = validation_csv_path
        self.img_dir_path = img_dir_path
        self.tokenizer = tokenizer
        self.input_dim = input_dim

        self.file_names, self.labels = self.__read_csv()
        self.transform = transforms.Compose([
                            transforms.ConvertImageDtype(torch.float),
                            transforms.CenterCrop(input_dim),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
                        ])

    def __read_csv(self):
        file_names = np.array([])
        labels = np.array([])
        with open(self.validation_csv_path, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in spamreader:
                ext = row[0][-4:]
                if ext in [".png",".jpe"]:
                    file_name = row[0].split("/")[-1]
                    file_label = row[1]

                    file_names = np.append(file_names, file_name)
                    labels = np.append(labels, file_label)

        return file_names, labels
    
    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir_path, (self.file_names[index]))
        image = read_image(img_path)
        
        #RESIZE, RESHAPE
        new_width = 150
        original_height, original_width = image.shape[1], image.shape[2]
        new_height = int((original_height / original_width) * new_width)
        resize_transform = transforms.Resize((new_height, new_width))
        resized_image = resize_transform(image)
        image = self.transform(resized_image)

        label = self.labels[index]
        label = self.tokenizer.encode(label)
        label = torch.from_numpy(label)

        return image, label
    
    def __len__(self,):
        return len(self.file_names)
    



#     import albumentations as A
# from albumentations.pytorch import ToTensorV2

# train_dataset.transform_A = A.Compose([
#     # Geometric & perspective-like distortions
#     A.ShiftScaleRotate(
#         shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5
#     ),
#     A.Perspective(
#         scale=(0.05, 0.1), p=0.3
#     ),

#     # Lighting & color variations
#     A.RandomBrightnessContrast(
#         brightness_limit=0.2, contrast_limit=0.2, p=0.3
#     ),
#     A.HueSaturationValue(
#         hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=10, p=0.3
#     ),
    
#     # Focus & blur adjustments
#     A.OneOf([
#         A.MotionBlur(blur_limit=3),
#         A.GaussianBlur(blur_limit=3),
#     ], p=0.3),

#     # Noise & grain
#     A.OneOf([
#         A.GaussNoise(var_limit=(10.0, 50.0)),
#         A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3))
#     ], p=0.3),

#     # # Partial occlusions using CoarseDropout
#     A.CoarseDropout(
#         max_holes=1, 
#         max_height=7, 
#         max_width=7, 
#         p=0.3
#     ),

#     # # Normalize and convert to Tensor
#     A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#     ToTensorV2()
# ])