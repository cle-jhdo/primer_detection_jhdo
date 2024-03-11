import os
import torch
import torch.nn as nn
import configuration_encoder

class PrimerDectionLoss(nn.Module):
    def __init__(self, config):
        super(PrimerDectionLoss, self).__init__()
        self.config = config
        self.mse_loss = torch.nn.MSELoss(reduction = 'none')
        self.BCE_loss = torch.nn.BCELoss(reduce='none')
        self.ref_loss = 0.5


    def set_ref_loss(self, ref_loss):
        self.ref_loss = ref_loss

    
    def modifiedMSE(self, input, target):
        mse = self.mse_loss(input, target)
        mse = torch.mean(mse, (1, 2, 3))
        mse_loss_vanila = mse.mean()
        weight = 2 * torch.sigmoid(2 * mse / self.ref_loss - 2)
        weight = self.config.loss_gamma ** weight
        mse_loss = mse * weight
        mse_loss = mse_loss.mean() / self.config.loss_gamma
        mse_loss = mse_loss - self.config.mse_loss_damp
        mse_loss = torch.clamp(mse_loss, min=0)
        
        return mse_loss, mse_loss_vanila


    def binaryCrossEntropyFocalLoss(self, input, target):
        BCE_loss = self.BCE_loss(input, target)
        pt = torch.exp(-BCE_loss)
        focal_loss = self.config.focal_loss_alpha * (1 - pt) ** self.config.focal_loss_gamma * BCE_loss

        return focal_loss.mean()
