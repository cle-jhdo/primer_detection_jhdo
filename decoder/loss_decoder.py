import os
import torch
import torch.nn as nn
import configuration_encoder

class PrimerDectionLoss(nn.Module):
    def __init__(self, config):
        super(PrimerDectionLoss, self).__init__()
        self.config = config
        self.mse_loss = torch.nn.MSELoss(reduction = 'none')
        self.ref_loss = 0.5


    def set_ref_loss(self, ref_loss):
        self.ref_loss = ref_loss

    # emphasize high-loss sample to solve data unbalance issue
    def modifiedMSE(self, input, target):
        mse = self.mse_loss(input, target)
        mse = torch.mean(mse, (1, 2, 3))
        mse_loss_vanila = mse.mean()
        weight = 2 * torch.sigmoid(2 * mse / self.ref_loss - 2)
        weight = self.config.loss_gamma ** weight
        mse_loss = mse * weight
        mse_loss = mse_loss.mean() / self.config.loss_gamma
        
        return mse_loss, mse_loss_vanila


