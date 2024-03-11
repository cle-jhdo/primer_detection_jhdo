import torch

class Configuration():
    def __init__(self, config_name = ''):
        # default values
        self.default_setup = {}
        self.lr = 1e-4
        self.lr_decay = 0.97
        self.load_checkpoint_epoch = 0
        self.unet_n_filters = [64, 128, 256, 512, 1024]

        self.batch_size = 10
        self.num_epoch = 500
        self.img_size = (256, 256) # h, w
        self.loss_gamma = 3
        self.brightness_augment = 0.3
        self.cutmix_flag = False
        self.cutmix_area = 0.4
        self.cutmix_p = 0.5
        self.dilation = [3, 3, 3, 3, 3]
        self.dilation_n_filters = [256, 256, 256, 256, 256]
        self.primer_score_threshold = 0.05 # 5% area of image
        self.primer_score_loss_weight = 0.1
        self.focal_loss_alpha = 0.5
        self.focal_loss_gamma = 2
        self.discriminator_score_loss_weight = 0.1
        self.disc_img_loss_weight = 0.1
        self.mse_loss_damp = 0.01

        #self.data_dir_train = '/root/Data/jhdo_dataset/train'
        #self.data_dir_valid = '/root/Data/jhdo_dataset/valid'
        self.data_dir_train = 'C:/Users/jhdo/project/jhdo_dataset/train'
        self.data_dir_valid = 'C:/Users/jhdo/project/jhdo_dataset/valid'
        self.check_point_dir = './checkpoint'
        self.log_dir = './log'


        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')