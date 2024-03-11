import sys
import numpy as np
import torch 
import os
import dataset_encoder
import encoder_v3
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


def save_gen_model(check_point_dir, model, optim,epoch):
    if not os.path.exists(check_point_dir):
        os.makedirs(check_point_dir)

    torch.save({'net':model.state_dict(),'optim':optim.state_dict()},'%s/model_gen_epoch%d.pt'%(check_point_dir, epoch))

def save_disc_model(check_point_dir, model, optim,epoch):
    if not os.path.exists(check_point_dir):
        os.makedirs(check_point_dir)

    torch.save({'net':model.state_dict(),'optim':optim.state_dict()},'%s/model_disc_epoch%d.pt'%(check_point_dir, epoch))


dataset_train = dataset_encoder.Dataset_v3(data_dir=config.data_dir_train, config = config)
loader_train = torch.utils.data.DataLoader(dataset_train, batch_size = config.batch_size, shuffle=True)

dataset_validation = dataset_encoder.Dataset_v3(data_dir=config.data_dir_valid, config = config)
loader_validation = torch.utils.data.DataLoader(dataset_validation, batch_size = config.batch_size , shuffle=True)

model_gen = encoder_v3.Encoder(config).to(config.device)
model_disc = encoder_v3.Discriminator(config).to(config.device)

loss_class = loss_encoder.PrimerDectionLoss(config)

optim_gen = torch.optim.Adam(model_gen.parameters(), lr = config.lr)
optim_disc = torch.optim.Adam(model_disc.parameters(), lr = config.lr)

num_train = len(dataset_train)
num_validation = len(dataset_validation)

num_train_for_epoch = np.ceil(num_train/config.batch_size)
num_val_for_epoch = np.ceil(num_validation/config.batch_size)

# Tensorbord
writer_train = SummaryWriter(log_dir=config.log_dir)
writer_val = SummaryWriter(log_dir=config.log_dir)

start_epoch = 0

if os.path.isfile(config.check_point_dir + '/' + 'model_gen_epoch' + str(config.load_checkpoint_epoch) + '.pt'):
    dict_model = torch.load(config.check_point_dir + '/' + 'model_gen_epoch' + str(config.load_checkpoint_epoch) + '.pt')
    model_gen.load_state_dict(dict_model['net'])
    optim_gen.load_state_dict(dict_model['optim'])
    start_epoch = config.load_checkpoint_epoch + 1 

if os.path.isfile(config.check_point_dir + '/' + 'model_disc_epoch' + str(config.load_checkpoint_epoch) + '.pt'):
    dict_model = torch.load(config.check_point_dir + '/' + 'model_disc_epoch' + str(config.load_checkpoint_epoch) + '.pt')
    model_disc.load_state_dict(dict_model['net'])
    optim_disc.load_state_dict(dict_model['optim'])

torch.autograd.set_detect_anomaly(True)
for epoch in range(start_epoch, config.num_epoch):
    model_gen.train()
    model_disc.train()
    optim_gen.param_groups[0]['lr'] = config.lr * (config.lr_decay ** (epoch / 5))
    optim_disc.param_groups[0]['lr'] = config.lr * (config.lr_decay ** (epoch / 5))

    loss_gen_arr = []
    loss_disc_arr = []
    loss_mse_arr = []
    loss_primer_score_arr = []
    print('Learning rate')
    print(optim_gen.param_groups[0]['lr'])

    batch_pick = int(random.random() * (num_train_for_epoch -1))
    for batch, data in enumerate(loader_train):
        img = data['img']
        img_label = data['img_label']
        is_primer = data['is_primer']
    
        img = img.to(config.device)
        img_label = img_label.to(config.device)
        is_primer = is_primer.to(config.device)
        restored_img = model_gen(img)
        primer_score, disc_score_origin = model_disc(img)
        #print('primer_disc')
        #print(is_primer)
        #print(primer_score)

        # backward
        optim_gen.zero_grad()
        optim_disc.zero_grad()
        primer_score_from_gen, discriminate_score_gen = model_disc(restored_img)
        mse_loss, mse_loss_vanila = loss_class.modifiedMSE(restored_img, img_label)
        loss_generator = -config.primer_score_loss_weight * torch.log(1 - primer_score_from_gen) - config.discriminator_score_loss_weight * torch.log(discriminate_score_gen)
        loss_generator = torch.mean(loss_generator)
        loss_generator = mse_loss + loss_generator
        loss_generator.backward(retain_graph=True)

        loss_discriminator = loss_class.binaryCrossEntropyFocalLoss(primer_score, is_primer)
        loss_discriminator = loss_discriminator - config.disc_img_loss_weight*torch.log(disc_score_origin) - config.disc_img_loss_weight*torch.log(1 - discriminate_score_gen)
        loss_discriminator = torch.mean(loss_discriminator)
        loss_discriminator.backward()
        optim_gen.step()
        optim_disc.step()

        # save loss
        loss_gen_arr += [loss_generator.item()]
        loss_disc_arr += [loss_discriminator.item()]
        loss_mse_arr += [mse_loss.item()]
        loss_primer_score_arr += [torch.mean(primer_score_from_gen).item()]

        print('Train Loss Mse ' + str(np.mean(loss_mse_arr)))
        print('Train Loss Primer score ' + str(np.mean(loss_primer_score_arr)))
        print('Train Loss tot ' + str(np.mean(loss_gen_arr)) + ' ' + str(np.mean(loss_disc_arr)) + ' ' + str(epoch))

        loss_class.set_ref_loss(np.mean(loss_gen_arr))
        restored_img = torch.clamp(restored_img, min = 0, max = 1)

        img_diff = torch.abs(restored_img - img_label)
        if batch == batch_pick and epoch%5 ==0:
            writer_train.add_image('train/label', torchvision.utils.make_grid(img_label), epoch)
            writer_train.add_image('train/input', torchvision.utils.make_grid(img), epoch)
            writer_train.add_image('train/restored', torchvision.utils.make_grid(restored_img), epoch)
            writer_train.add_image('train/img_diff', torchvision.utils.make_grid(img_diff), epoch)

    writer_train.add_scalar('train/loss_gen', np.mean(loss_gen_arr), epoch)
    writer_train.add_scalar('train/loss_disc', np.mean(loss_disc_arr), epoch)
    writer_train.add_scalar('train/loss_mse', np.mean(loss_mse_arr), epoch)
    writer_train.add_scalar('train/loss_primer_score', np.mean(loss_primer_score_arr), epoch)
   
    # validation
    if epoch%5 == 0:
        batch_pick = int(random.random() * (num_val_for_epoch -1))
        with torch.no_grad():
            model_gen.eval()
            model_disc.eval()
            loss_mse_arr = []
            loss_disc_arr = []
            loss_disc_gen_arr = []
            loss_primer_score_arr = []

            for batch, data in enumerate(loader_validation):
                img = data['img']
                img_label = data['img_label']
                is_primer = data['is_primer']
            
                img = img.to(config.device)
                img_label = img_label.to(config.device)
                is_primer = is_primer.to(config.device)
                restored_img = model_gen(img)
                primer_score, disc_score = model_disc(img)
                primer_score_gen, disc_score_gen = model_disc(restored_img)

                loss_primer_score = loss_class.binaryCrossEntropyFocalLoss(primer_score, is_primer)
                mse_loss, mse_loss_vanila = loss_class.modifiedMSE(restored_img, img_label)
                loss_discriminator = -torch.log(disc_score)
                loss_discriminator_gen = -torch.log(1 - disc_score_gen)

                loss_mse_arr += [mse_loss_vanila.item()]
                loss_disc_arr += [torch.mean(loss_discriminator).item()]
                loss_disc_gen_arr += [torch.mean(loss_discriminator_gen).item()]
                loss_primer_score_arr += [torch.mean(loss_primer_score).item()]

                print('Val Loss Mse ' + str(np.mean(loss_mse_arr)))
                print('Val Loss Primer score ' + str(np.mean(loss_primer_score_arr)))
                print('Val Loss discriminator origin ' + str(np.mean(loss_gen_arr)) + ' ' + str(epoch))
                print('Val Loss discriminator fake ' + str(np.mean(loss_disc_gen_arr)) + ' ' + str(epoch))

                restored_img = torch.clamp(restored_img, min = 0, max = 1)
                img_diff = torch.abs(restored_img - img_label)
                if batch == batch_pick:
                    writer_val.add_image('valid/label', torchvision.utils.make_grid(img_label), epoch)
                    writer_val.add_image('valid/input', torchvision.utils.make_grid(img), epoch)
                    writer_val.add_image('valid/restored', torchvision.utils.make_grid(restored_img), epoch)
                    writer_val.add_image('valid/img_diff', torchvision.utils.make_grid(img_diff), epoch)

            writer_train.add_scalar('valid/loss_mse', np.mean(loss_gen_arr), epoch)
            writer_train.add_scalar('valid/loss_primer_score', np.mean(loss_disc_arr), epoch)
            writer_train.add_scalar('valid/discriminator_origin', np.mean(loss_disc_arr), epoch)
            writer_train.add_scalar('valid/loss_primer_score', np.mean(loss_disc_gen_arr), epoch)

            save_gen_model(check_point_dir=config.check_point_dir, model = model_gen, optim = optim_gen, epoch = epoch)
            save_disc_model(check_point_dir=config.check_point_dir, model = model_disc, optim = optim_disc, epoch = epoch)

writer_train.close()
writer_val.close()