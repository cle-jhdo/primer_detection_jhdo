import os
import numpy as np
import torch
import torch.nn as nn
import configuration_encoder

class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.sigmoid = torch.nn.Sigmoid()

        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, dilation = 1):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding, dilation = dilation,
                                 bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ELU()]

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

        self.enc4_dilate = CBR2d(in_channels=config.unet_n_filters[3], out_channels=config.dilation_n_filters[3], dilation=config.dilation[3], padding='same')
        self.enc4_1x1 = CBR2d(in_channels=config.dilation_n_filters[3], out_channels=config.dilation_n_filters[3]//2, kernel_size=1, padding=0)

        self.enc5_1 = CBR2d(in_channels=config.unet_n_filters[3] + config.dilation_n_filters[3]//2, out_channels=config.unet_n_filters[4])

        self.enc5_dilate1 = CBR2d(in_channels=config.unet_n_filters[4], out_channels=config.dilation_n_filters[4], dilation=config.dilation[4], padding=0)
        self.enc5_dilate2 = CBR2d(in_channels=config.dilation_n_filters[4], out_channels=config.dilation_n_filters[4], dilation=config.dilation[4], padding=0)
        self.enc5_1x1 = CBR2d(in_channels=config.dilation_n_filters[4] * 2, out_channels=config.dilation_n_filters[4], kernel_size=1, padding=0)

        self.dec_prediction1 = CBR2d(in_channels=config.unet_n_filters[4] + config.dilation_n_filters[4], out_channels=config.unet_n_filters[4]//16, kernel_size=1, padding=0)
        self.dec_prediction2 = CBR2d(in_channels=config.unet_n_filters[4]//16, out_channels=config.unet_n_filters[4]//128, kernel_size=1, padding=0)

        self.dec_discriminate1 = CBR2d(in_channels=config.unet_n_filters[4] + config.dilation_n_filters[4], out_channels=config.unet_n_filters[4]//16, kernel_size=1, padding=0)
        self.dec_discriminate2 = CBR2d(in_channels=config.unet_n_filters[4]//16, out_channels=config.unet_n_filters[4]//128, kernel_size=1, padding=0)

        self.prediction = torch.nn.Conv2d(in_channels=config.unet_n_filters[4]//128, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)
        self.discriminate = torch.nn.Conv2d(in_channels=config.unet_n_filters[4]//128, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

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

        enc4_dilated = self.enc4_dilate(pool4)
        enc4_1x1 = self.enc4_1x1(enc4_dilated)
        enc4_3 = torch.cat((pool4, enc4_1x1), dim = 1)

        enc5_1 = self.enc5_1(enc4_3)
        enc5_dilated1 = self.enc5_dilate1(enc5_1)
        enc5_dilated2 = self.enc5_dilate2(enc5_dilated1)
        enc5_dilated2_upsample = torch.nn.functional.interpolate(enc5_dilated2, (enc5_dilated1.shape[2], enc5_dilated1.shape[3]), mode = 'bilinear')
        enc5_dilate2_cat = torch.cat((enc5_dilated1, enc5_dilated2_upsample), dim = 1)
        enc5_1x1 = self.enc5_1x1(enc5_dilate2_cat)
        enc5_1x1_upsample = torch.nn.functional.interpolate(enc5_1x1, (enc5_1.shape[2], enc5_1.shape[3]), mode = 'bilinear')
        enc5_2 = torch.cat((enc5_1, enc5_1x1_upsample), dim = 1)
        
        dec_prediction1 = self.dec_prediction1(enc5_2)
        dec_prediction2 = self.dec_prediction2(dec_prediction1)

        dec_disc1 = self.dec_discriminate1(enc5_2)
        dec_disc2 = self.dec_discriminate2(dec_disc1)

        primer_score = self.prediction(dec_prediction2)
        primer_score = torch.mean(primer_score, dim = (2, 3))
        primer_score = self.sigmoid(primer_score)

        disc_score = self.discriminate(dec_disc2)
        disc_score = torch.mean(disc_score, dim = (2, 3))
        disc_score = self.sigmoid(disc_score)

        return primer_score, disc_score

        
class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()

        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, dilation = 1):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding, dilation = dilation,
                                 bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ELU()]

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

        self.enc4_dilate = CBR2d(in_channels=config.unet_n_filters[3], out_channels=config.dilation_n_filters[3], dilation=config.dilation[3], padding='same')
        self.enc4_1x1 = CBR2d(in_channels=config.dilation_n_filters[3], out_channels=config.dilation_n_filters[3]//2, kernel_size=1, padding=0)

        self.enc5_1 = CBR2d(in_channels=config.unet_n_filters[3] + config.dilation_n_filters[3]//2, out_channels=config.unet_n_filters[4])

        self.enc5_dilate1 = CBR2d(in_channels=config.unet_n_filters[4], out_channels=config.dilation_n_filters[4], dilation=config.dilation[4], padding=0)
        self.enc5_dilate2 = CBR2d(in_channels=config.dilation_n_filters[4], out_channels=config.dilation_n_filters[4], dilation=config.dilation[4], padding=0)
        self.enc5_1x1 = CBR2d(in_channels=config.dilation_n_filters[4] * 2, out_channels=config.dilation_n_filters[4], kernel_size=1, padding=0)

        # Expansive path
        self.dec5_1 = CBR2d(in_channels=config.unet_n_filters[4] + config.dilation_n_filters[4], out_channels=config.unet_n_filters[3])

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

        enc4_dilated = self.enc4_dilate(pool4)
        enc4_1x1 = self.enc4_1x1(enc4_dilated)
        enc4_3 = torch.cat((pool4, enc4_1x1), dim = 1)

        enc5_1 = self.enc5_1(enc4_3)
        enc5_dilated1 = self.enc5_dilate1(enc5_1)
        enc5_dilated2 = self.enc5_dilate2(enc5_dilated1)
        enc5_dilated2_upsample = torch.nn.functional.interpolate(enc5_dilated2, (enc5_dilated1.shape[2], enc5_dilated1.shape[3]), mode = 'bilinear')
        enc5_dilate2_cat = torch.cat((enc5_dilated1, enc5_dilated2_upsample), dim = 1)
        enc5_1x1 = self.enc5_1x1(enc5_dilate2_cat)
        enc5_1x1_upsample = torch.nn.functional.interpolate(enc5_1x1, (enc5_1.shape[2], enc5_1.shape[3]), mode = 'bilinear')
        enc5_2 = torch.cat((enc5_1, enc5_1x1_upsample), dim = 1)

        dec5_1 = self.dec5_1(enc5_2)

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