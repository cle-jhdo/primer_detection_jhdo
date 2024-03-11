import torch
import torchvision
import numpy as np
import cv2
import random
import configuration_decoder
import albumentations
import albumentations.pytorch

class PreProcess():
    def __init__(self, config):
        self.config = config


    def gammaCorrection(self, img, gamma):
        invGamma = 1 / gamma

        table = [((i / 255) ** invGamma) * 255 for i in range(256)]
        table = np.array(table, np.uint8)

        return cv2.LUT(img, table)
            
        
    def augmentation(self, img, img_label):
        albumentations_transform = albumentations.Compose([
            albumentations.OneOf([
                albumentations.Compose
                ([
                    albumentations.Resize(300 + int(random.random() * 100), 300 + int(random.random() * 100)), 
                    albumentations.RandomCrop(self.config.img_size[0], self.config.img_size[1]),
                ], p = 1),
                
                albumentations.Resize(self.config.img_size[0], self.config.img_size[1], p = 1)
            ], p = 1),
            
            albumentations.OneOf([
                albumentations.HorizontalFlip(p=0.5),
                albumentations.Rotate(p=1),
                albumentations.VerticalFlip(p=0.5)
            ], p=1)
            
        ],  additional_targets={'image_label': 'image'})

        color_jitter = albumentations.ColorJitter(brightness = self.config.brightness_augment, hue = 0, saturation = 0, p = 1)

        img_transformed = albumentations_transform(image = img, image_label = img_label)
        img_aug = img_transformed['image']
        img_label_aug = img_transformed['image_label']
        img_aug = color_jitter(image = img_aug)['image']

        to_tensor = albumentations.pytorch.ToTensorV2()
        img_aug = to_tensor(image = img_aug)['image']
        img_label_aug = to_tensor(image = img_label_aug)['image']

        return img_aug, img_label_aug