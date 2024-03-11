import os
import numpy as np
import torch
import torch.nn as nn
import configuration_encoder

class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()

        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ReLU()]

            cbr = nn.Sequential(*layers)

            return cbr

        # Contracting path
        self.enc1_1 = CBR2d(in_channels=3, out_channels=config.unet_n_filters[0])
        self.enc1_2 = CBR2d(in_channels=config.unet_n_filters[0], out_channels=config.unet_n_filters[0])

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc2_1 = CBR2d(in_channels=config.unet_n_filters[0], out_channels=config.unet_n_filters[1])
        self.enc2_2 = CBR2d(in_channels=config.unet_n_filters[1], out_channels=config.unet_n_filters[1])

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3_1 = CBR2d(in_channels=config.unet_n_filters[1], out_channels=config.unet_n_filters[2])
        self.enc3_2 = CBR2d(in_channels=config.unet_n_filters[2], out_channels=config.unet_n_filters[2])

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.enc4_1 = CBR2d(in_channels=config.unet_n_filters[2], out_channels=config.unet_n_filters[3])
        self.enc4_2 = CBR2d(in_channels=config.unet_n_filters[3], out_channels=config.unet_n_filters[3])

        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.enc5_1 = CBR2d(in_channels=config.unet_n_filters[3], out_channels=config.unet_n_filters[4])

        # Expansive path
        self.dec5_1 = CBR2d(in_channels=config.unet_n_filters[4], out_channels=config.unet_n_filters[3])

        self.unpool4 = nn.ConvTranspose2d(in_channels=config.unet_n_filters[3], out_channels=config.unet_n_filters[3],
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec4_2 = CBR2d(in_channels=2 * config.unet_n_filters[3], out_channels=config.unet_n_filters[3])
        self.dec4_1 = CBR2d(in_channels=config.unet_n_filters[3], out_channels=config.unet_n_filters[2])

        self.unpool3 = nn.ConvTranspose2d(in_channels=config.unet_n_filters[2], out_channels=config.unet_n_filters[2],
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec3_2 = CBR2d(in_channels=2 * config.unet_n_filters[2], out_channels=config.unet_n_filters[2])
        self.dec3_1 = CBR2d(in_channels=config.unet_n_filters[2], out_channels=config.unet_n_filters[1])

        self.unpool2 = nn.ConvTranspose2d(in_channels=config.unet_n_filters[1], out_channels=config.unet_n_filters[1],
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec2_2 = CBR2d(in_channels=2 * config.unet_n_filters[1], out_channels=config.unet_n_filters[1])
        self.dec2_1 = CBR2d(in_channels=config.unet_n_filters[1], out_channels=config.unet_n_filters[0])

        self.unpool1 = nn.ConvTranspose2d(in_channels=config.unet_n_filters[0], out_channels=config.unet_n_filters[0],
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec1_2 = CBR2d(in_channels=2 * config.unet_n_filters[0], out_channels=config.unet_n_filters[0])
        self.dec1_1 = CBR2d(in_channels=config.unet_n_filters[0], out_channels=config.unet_n_filters[0])

        self.fc_restoring = nn.Conv2d(in_channels=config.unet_n_filters[0], out_channels=3, kernel_size=1, stride=1, padding=0, bias=True)


    def forward(self, x):
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)

        enc5_1 = self.enc5_1(pool4)

        dec5_1 = self.dec5_1(enc5_1)

        unpool4 = self.unpool4(dec5_1)
        cat4 = torch.cat((unpool4, enc4_2), dim=1)
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)

        unpool3 = self.unpool3(dec4_1)
        cat3 = torch.cat((unpool3, enc3_2), dim=1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)

        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat((unpool2, enc2_2), dim=1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)

        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((unpool1, enc1_2), dim=1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        restoring = self.fc_restoring(dec1_1)

        return restoring