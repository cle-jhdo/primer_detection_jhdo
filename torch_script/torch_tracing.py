import torch
import torchvision
import sys
sys.path.insert(1, '../encoder')
sys.path.insert(2, '../decoder')
import encoder
import configuration_encoder
import configuration_decoder
import combined_model_v2

ptfile_encoder = 'model_epoch350.pt'
ptfile_decoder = 'model_decoder_epoch50.pt'

config_encoder = configuration_encoder.Configuration('unet_size_big')
config_decoder = configuration_decoder.Configuration('unet_size_big')

encoder_unet = encoder.Encoder(config_encoder)
decoder_unet = combined_model_v2.CombinedModel(config_decoder)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dict_model_decoder = torch.load(ptfile_decoder, map_location=torch.device(device=device))
decoder_unet.load_state_dict(dict_model_decoder['net'])
dict_model_encoder = torch.load(ptfile_encoder, map_location=torch.device(device=device))
encoder_unet.load_state_dict(dict_model_encoder['net'])

encoder_unet.eval()
decoder_unet.eval()
decoder_unet.set_encoder(encoder_unet)

example = torch.ones(1, 3, config_encoder.img_size[0], config_encoder.img_size[1])

traced_script_module_decoder = torch.jit.trace(decoder_unet, example)

print(traced_script_module_decoder.code)
traced_script_module_decoder.save("decoder_torchscript.pt")