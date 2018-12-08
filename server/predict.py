import os
import argparse
import torch
import json
import numpy
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image

from server.utils import poisson_blend, gen_input_mask
from server.models import CompletionNetwork


def generateMask(shape, img_w, img_h, x1, y1, x2, y2):
    mask = torch.zeros(shape)
    bsize, _, mask_h, mask_w = mask.shape
    masks = []
    new_h = img_h * 300 / img_w
    x1_norm = int(x1 / 300 * 160)
    y1_norm = int(y1 / new_h * 160)
    x2_norm = int(x2 / 300 * 160)
    y2_norm = int(y2 / new_h * 160)
    for i in range(bsize):
        mask[i, :, y1_norm : y2_norm, x1_norm : x2_norm] = 1.0
    return mask


# python predict.py model_cn_step400000 config.json input.jpg output.jpg
def generateNewFace(filename, type, x1, y1, x2, y2):
    output_filename = "dist/output.jpg"
    model_body = os.path.expanduser("server/model_cn_step40000")
    config = os.path.expanduser("server/config.json")
    input_img = os.path.expanduser(filename)
    output_img = os.path.expanduser("static/" + output_filename)
    max_holes = 1
    img_size = 160

    # =============================================
    # Load model
    # =============================================
    with open(config, 'r') as f:
        config = json.load(f)
    mpv = config['mean_pv']
    model = CompletionNetwork()
    model.load_state_dict(torch.load(model_body, map_location='cpu'))


    # =============================================
    # Predict
    # =============================================
    # convert img to tensor
    img = Image.open(input_img)
    img_w, img_h = img.size
    img = transforms.Resize(img_size)(img)
    # img = transforms.Randomrop((img_size, img_size))(img)
    x = transforms.ToTensor()(img)
    x = torch.unsqueeze(x, dim=0)
    if 'png' in type:
        x = x[:, 0:3, :, :]

    # create mask
    msk = generateMask(x.shape, img_w, img_h, int(x1), int(y1), int(x2), int(y2))

    # inpaint
    with torch.no_grad():
        input = x - x * msk + mpv * msk
        output = model(input)
        inpainted = poisson_blend(input, output, msk)
        imgs = torch.cat((x, input, inpainted), dim=-1)
        imgs = save_image(imgs, output_img, nrow=3)
    print('output img was saved as %s.' % output_img)
    return output_filename
