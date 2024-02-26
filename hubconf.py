dependencies = ['torch', "timm"]

import torch
import swin

MODEL_URLS = {
    "swinv2_large_patch4_window12_192_22k": "https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_large_patch4_window12_192_22k.pth"
}

def swinv2_large_patch4_window12_192_22k(pretrained=False):
    model = swin.SwinTransformer()
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(MODEL_URLS["swinv2_large_patch4_window12_192_22k"], map_location="cpu")['model'])
    return model
