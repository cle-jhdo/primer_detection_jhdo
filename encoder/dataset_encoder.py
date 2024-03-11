import numpy as np
import torch
import os
import pre_processing_encoder
import cv2
import random

def get_files_recursive(folder_path):
    file_list = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if 'truth' in file:
                continue
            if file.split('.')[-1].lower() not in ['png', 'jpg', 'bmp']:
                continue
            file_path = os.path.abspath(os.path.join(root, file))
            file_list.append(file_path)
        for subdir in dirs:
            subdir_path = os.path.join(root, subdir)
            file_list.extend(get_files_recursive(subdir_path))
    
    return file_list


def get_files_recursive_with_ground_truth(folder_path):
    file_dict = {}
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.split('.')[-1].lower() not in ['png', 'jpg', 'bmp']:
                continue
            if 'truth' in file:
                continue
            file_path = os.path.abspath(os.path.join(root, file))
            file_path_ground_truth = file_path.split('.')[0] + '_ground_truth.' + file_path.split('.')[1]
            if os.path.isfile(file_path_ground_truth) == True:
                file_dict[file_path] = file_path_ground_truth
            else:
                file_dict[file_path] = ''
        for subdir in dirs:
            subdir_path = os.path.join(root, subdir)
            file_dict.update(get_files_recursive_with_ground_truth(subdir_path))
    
    return file_dict


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, config):
        self.data_dir = data_dir
        self.pre_process = pre_processing_encoder.PreProcess(config)
        self.data_list = get_files_recursive(data_dir)
        self.data_list_synth = ''
    	        

    def __len__(self):
        return len(self.data_list)
	
    
    def __getitem__(self, index):
        #img = cv2.imread(self.data_list[index])
        img_array = np.fromfile(self.data_list[index], np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img, img_flip = self.pre_process.augmentation(img)  
        img = img/255.0
        img_flip = img_flip/255.0
        
        data = {'img':img, 'img_label':img_flip}

        return data


    def set_synth_dir(self, sysnth_dir):
        self.data_list_synth = self.get_files_recursive(sysnth_dir)


class Dataset_v3(torch.utils.data.Dataset):
    def __init__(self, data_dir, config):
        self.data_dir = data_dir
        self.pre_process = pre_processing_encoder.PreProcess(config)
        self.data_dict = get_files_recursive_with_ground_truth(data_dir)
        self.config = config


    def __len__(self):
        return len(self.data_dict.keys())
	

    def __getitem__(self, index):
        img_file = list(self.data_dict.keys())[index]
        img_label_file = self.data_dict[img_file]
        img_array = np.fromfile(img_file, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img, img_flip = self.pre_process.augmentation(img)
        img = img/255.0
        img_flip = img_flip/255.0
        img_label = torch.zeros((1, self.config.img_size[1], self.config.img_size[0]))
        if len(img_label_file) > 0:
            img_label_array = np.fromfile(img_label_file, np.uint8)
            img_label = cv2.imdecode(img_label_array, cv2.IMREAD_COLOR)
            img_label = cv2.cvtColor(img_label, cv2.COLOR_BGR2GRAY)
            img_label = cv2.resize(img_label, self.config.img_size)
            img_label = torch.from_numpy(np.array(img_label, dtype=np.float32))
            img_label = img_label / 255
        
        img_label_score = torch.mean(img_label)
        img_label_score = (img_label_score - self.config.primer_score_threshold) / self.config.primer_score_threshold
        img_label_score = torch.sigmoid(img_label_score)
        img_label_score = torch.unsqueeze(img_label_score, dim = 0)


        #is_primer = torch.tensor([0.0])
        #if img_label_score.item() > self.config.primer_score_threshold:
        #    is_primer = torch.tensor([1.0])

        data = {'img':img, 'img_label':img_flip, 'is_primer':img_label_score}

        return data