import numpy as np
import torch
import os
import pre_processing_decoder
import cv2
import random

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, config):
        self.data_dir = data_dir
        self.pre_process = pre_processing_decoder.PreProcess(config)
        self.data_dict = self.get_files_recursive(data_dir)

    def __len__(self):
        return len(self.data_dict.keys())
	
    def __getitem__(self, index):
        img_file = list(self.data_dict.keys())[index]
        img_label_file = self.data_dict[img_file]
        img_array = np.fromfile(img_file, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img_label_array = np.fromfile(img_label_file, np.uint8)
        img_label = cv2.imdecode(img_label_array, cv2.IMREAD_COLOR)
        img_label = cv2.cvtColor(img_label, cv2.COLOR_BGR2GRAY)
        img_aug, img_label_aug = self.pre_process.augmentation(img, img_label)
        img_aug = img_aug/255.0
        img_label_aug = img_label_aug/255.0
        
        data = {'img':img_aug, 'img_label':img_label_aug}

        return data


    def get_files_recursive(self, folder_path):
        file_dict = {}
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.split('.')[-1].lower() not in ['png', 'jpg', 'bmp']:
                    continue
                
                file_path = os.path.abspath(os.path.join(root, file))
                file_path_ground_truth = file_path.split('.')[0] + '_ground_truth.' + file_path.split('.')[1]
                if os.path.isfile(file_path_ground_truth) == True:
                    file_dict[file_path] = file_path_ground_truth
            for subdir in dirs:
                subdir_path = os.path.join(root, subdir)
                file_dict.update(self.get_files_recursive(subdir_path))
        
        return file_dict