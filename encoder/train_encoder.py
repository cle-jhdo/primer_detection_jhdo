import sys
import numpy as np
import torch 
import os
import dataset_encoder
import encoder
import configuration_encoder
import random
import torchvision
import loss_encoder
from torch.utils.tensorboard import SummaryWriter

config_name = ''
if len(sys.argv) == 2:
    config_name = sys.argv[1]

config = configuration_encoder.Configuration(config_name)

if not os.path.exists(config.log_dir):
    os.makedirs(config.log_dir)

vertical_flip = torchvision.transforms.RandomVerticalFlip(p=1)


def rand_bbox(height, width, box_area):
    cut_rat = np.sqrt(1. - box_area)
    cut_w = int(width * cut_rat)
    cut_h = int(height * cut_rat)

    # uniform
    cx = np.random.randint(width)
    cy = np.random.randint(height)

    bbx1 = np.clip(cx - cut_w // 2, 0, width)
    bby1 = np.clip(cy - cut_h // 2, 0, height)
    bbx2 = np.clip(cx + cut_w // 2, 0, width)
    bby2 = np.clip(cy + cut_h // 2, 0, height)

    return bbx1, bby1, bbx2, bby2


def cutmix_img(img, img_mix):
    mix_size = 0.1 + random.random() * config.cutmix_area
    bbx1, bby1, bbx2, bby2 = rand_bbox(img.shape[1], img.shape[2], mix_size)
    img[:, bbx1:bbx2, bby1:bby2] = img_mix[:, bbx1:bbx2, bby1:bby2]

    return img


def cutmix_batch(batch_img):
    cutmix_batch = []
    cutmix_batch_flip = []
    for i in range(batch_img.shape[0]):
        img_tmp = batch_img[i]
        if random.random() > 0.5:
            random_idx = int(random.random()*batch_img.shape[0])
            if random_idx != i:
                img_mix = batch_img[random_idx]
                img_tmp = cutmix_img(img_tmp, img_mix)
        
        cutmix_batch.append(img_tmp.unsqueeze(0))
        img_tmp_flip = vertical_flip(img_tmp)
        cutmix_batch_flip.append(img_tmp_flip.unsqueeze(0))

    cutmix_batch = torch.cat(cutmix_batch, dim = 0)
    cutmix_batch_flip = torch.cat(cutmix_batch_flip, dim = 0)

    return cutmix_batch, cutmix_batch_flip


def save(check_point_dir, model, optim,epoch):
    if not os.path.exists(check_point_dir):
        os.makedirs(check_point_dir)

    torch.save({'net':model.state_dict(),'optim':optim.state_dict()},'%s/model_epoch%d.pt'%(check_point_dir, epoch))


dataset_train = dataset_encoder.Dataset(data_dir=config.data_dir_train, config = config)
loader_train = torch.utils.data.DataLoader(dataset_train, batch_size = config.batch_size, shuffle=True)

dataset_validation = dataset_encoder.Dataset(data_dir=config.data_dir_valid, config = config)
loader_validation = torch.utils.data.DataLoader(dataset_validation, batch_size = config.batch_size , shuffle=True)

model = encoder.Encoder(config).to(config.device)

loss_ptr = loss_encoder.PrimerDectionLoss(config)

optim = torch.optim.Adam(model.parameters(), lr = config.lr)

num_train = len(dataset_train)
num_validation = len(dataset_validation)

num_train_for_epoch = np.ceil(num_train/config.batch_size)
num_val_for_epoch = np.ceil(num_validation/config.batch_size)

# Tensorbord
writer_train = SummaryWriter(log_dir=config.log_dir)
writer_val = SummaryWriter(log_dir=config.log_dir)

start_epoch = 0

if os.path.isfile(config.check_point_dir + '/' + 'model_epoch' + str(config.load_checkpoint_epoch) + '.pt'):
    dict_model = torch.load(config.check_point_dir + '/' + 'model_epoch' + str(config.load_checkpoint_epoch) + '.pt')
    model.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    start_epoch = config.load_checkpoint_epoch + 1 

for epoch in range(start_epoch, config.num_epoch):
    model.train()
    loss_arr = []
    optim.param_groups[0]['lr'] = config.lr * (config.lr_decay ** (epoch / 5))
    print('Learning rate')
    print(optim.param_groups[0]['lr'])

    batch_pick = int(random.random() * (num_train_for_epoch -1))
    for batch, data in enumerate(loader_train):
        img = data['img']
        img_label = data['img_label']
        if config.cutmix_flag == True:
            img, img_label = cutmix_batch(img)
    
        img = img.to(config.device)
        img_label = img_label.to(config.device)
        restored_img = model(img)

        # backward
        optim.zero_grad() 
        mse_loss, mse_loss_vanila = loss_ptr.modifiedMSE(restored_img, img_label)
        train_loss = mse_loss
        train_loss.backward()
        optim.step()

        # save loss
        loss_arr += [train_loss.item()]

        print('Train Loss Mse ' + str(mse_loss_vanila.item()))
        print('Train Loss tot ' + str(np.mean(loss_arr)) + ' ' + str(epoch))

        loss_ptr.set_ref_loss(np.mean(loss_arr))
        restored_img = torch.clamp(restored_img, min = 0, max = 1)

        if batch == batch_pick and epoch%5 ==0:
            writer_train.add_image('train/label', torchvision.utils.make_grid(img_label), epoch)
            writer_train.add_image('train/input', torchvision.utils.make_grid(img), epoch)
            writer_train.add_image('train/restored', torchvision.utils.make_grid(restored_img), epoch)

    writer_train.add_scalar('train/loss', np.mean(loss_arr), epoch)
   
    # validation
    if epoch%5 == 0:
        batch_pick = int(random.random() * (num_val_for_epoch -1))
        with torch.no_grad():
            model.eval()
            loss_arr = []

            for batch, data in enumerate(loader_validation):
                img = data['img'].to(config.device)
                img_label = data['img_label'].to(config.device)

                restored_img = model(img)
        
                mse_loss, mse_loss_vanila = loss_ptr.modifiedMSE(restored_img, img_label)
                valid_loss = mse_loss_vanila

                loss_arr += [valid_loss.item()]
                print('Val Loss Mse ' + str(mse_loss_vanila.item()))
                print('Val Loss tot ' + str(np.mean(loss_arr)) + ' ' + str(epoch))
                restored_img = torch.clamp(restored_img, min = 0, max = 1)
                
                if batch == batch_pick:
                    writer_val.add_image('valid/label', torchvision.utils.make_grid(img_label), epoch)
                    writer_val.add_image('valid/input', torchvision.utils.make_grid(img), epoch)
                    writer_val.add_image('valid/restored', torchvision.utils.make_grid(restored_img), epoch)

            writer_val.add_scalar('valid/loss', np.mean(loss_arr), epoch)
            save(check_point_dir=config.check_point_dir, model = model, optim = optim, epoch = epoch)

writer_train.close()
writer_val.close()