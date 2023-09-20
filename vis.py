from collections import OrderedDict

import torch
from layers.pretrained.basic_unet import BasicUNet, BasicUNetEncoder

weights = "pretrained/basic_unet/pretrained.pt"

state_dict = torch.load(weights)
keys = list(state_dict.keys())
ups_0 = next((i for i, key in enumerate(keys) if key.startswith('ups')), None)

encoder_state_dict = OrderedDict()
for i, (k, v) in enumerate(state_dict.items()):
    if i < ups_0:
        encoder_state_dict[k] = v
    else:
        break

print(type(encoder_state_dict))
torch.save(encoder_state_dict, "pretrained/basic_unet/encoder.pt")
# model = BasicUNetEncoder()
# model.load_state_dict(encoder_state_dict)