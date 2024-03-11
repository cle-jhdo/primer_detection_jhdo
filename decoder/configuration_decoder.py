import torch
import sys
sys.path.insert(1, '../encoder')

class Configuration():
    def __init__(self, config_name = ''):
        # default values
        self.default_setup = {}
        self.lr = 3e-4
        self.lr_decay = 0.97
        self.load_checkpoint_epoch = 0
        self.unet_n_filters = [64, 128, 256, 512, 1024]

        self.batch_size = 20
        self.num_epoch = 500
        self.img_size = (256, 256) # w, h
        self.loss_gamma = 3
        self.brightness_augment = 0.3
        self.cutmix_flag = False
        self.cutmix_area = 0.4
        self.cutmix_p = 0.5
        self.data_dir_train = '/root/Data/jhdo_dataset/train'
        self.data_dir_valid = '/root/Data/jhdo_dataset/valid'
        #self.data_dir_train = '../../jhdo_dataset/train'
        #self.data_dir_valid = '../../jhdo_dataset/valid'

        self.check_point_dir = './checkpoint'
        self.log_dir = './log'
        self.encoder_checkpoint_file = '../encoder/checkpoint_unet_size_big/model_epoch350.pt'
        self.dilation = 3
        self.dilation_n_filters = [256, 64]
        self.use_decoder_v2 = False

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')