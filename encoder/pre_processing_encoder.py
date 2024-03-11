import torch
import torchvision
import numpy as np
import cv2
import random
import configuration_encoder
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
            
        
    def augmentation(self, img):
        albumentations_transform = albumentations.Compose([
            albumentations.OneOf([
                albumentations.Compose
                ([
                    albumentations.Resize(300 + int(random.random() * 50), 300 + int(random.random() * 50)), 
                    albumentations.RandomCrop(self.config.img_size[0], self.config.img_size[1]),
                ], p = 1),
                
                albumentations.Resize(self.config.img_size[0], self.config.img_size[1], p = 1)
            ], p = 1),
            
            albumentations.OneOf([
                albumentations.HorizontalFlip(p=0.5),
                albumentations.Rotate(p=1),
                albumentations.VerticalFlip(p=0.5)
            ], p=1),

            albumentations.ColorJitter(brightness = self.config.brightness_augment, hue = 0, saturation = 0, p = 1),

            albumentations.pytorch.ToTensorV2()
        ])

        img_aug = albumentations_transform(image=img)['image']
        img_flip = torchvision.transforms.functional.vflip(img_aug)

        return img_aug, img_flip