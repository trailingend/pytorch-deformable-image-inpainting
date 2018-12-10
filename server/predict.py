import os
import argparse
import torch
import json
import numpy as np
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image

from server.utils import poisson_blend, gen_input_mask
from server.models import CompletionNetwork


# def generateMask(shape, img_w, img_h, x1, y1, x2, y2):
#     mask = torch.zeros(shape)
#     bsize, _, mask_h, mask_w = mask.shape
#     masks = []
#     new_h = img_h * 300 / img_w
#     x1_norm = int(x1 / 300 * 160)
#     y1_norm = int(y1 / new_h * 160)
#     x2_norm = int(x2 / 300 * 160)
#     y2_norm = int(y2 / new_h * 160)
#     for i in range(bsize):
#         mask[i, :, y1_norm : y2_norm, x1_norm : x2_norm] = 1.0
#     return mask
#
#
# # python predict.py model_cn_step400000 config.json input.jpg output.jpg
# def generateNewFace(filename, type, x1, y1, x2, y2):
#     output_filename = "dist/output.jpg"
#     model_body = os.path.expanduser("server/model_cn_step40000")
#     config = os.path.expanduser("server/config.json")
#     input_img = os.path.expanduser(filename)
#     output_img = os.path.expanduser("static/" + output_filename)
#     max_holes = 1
#     img_size = 160
#
#     # =============================================
#     # Load model
#     # =============================================
#     with open(config, 'r') as f:
#         config = json.load(f)
#     mpv = config['mean_pv']
#     model = CompletionNetwork()
#     model.load_state_dict(torch.load(model_body, map_location='cpu'))


def generateMask(shape, img_w, img_h, maskname):
    mask = Image.open(maskname)
    mask = mask.resize((img_w, img_h))

    mask_rd = np.asarray(mask)
    mask_np = np.copy(mask_rd)
    mask_np = mask_np[:, :, 0:3]
    mask_np[mask_np > 0] = 1
    mask_np = np.moveaxis(mask_np, -1, 0)
    mask_full = np.expand_dims(mask_np, axis=0)

    msk = torch.from_numpy(mask_full)
    msk = msk.float()
    return msk


# python predict.py model_cn_step400000 config.json input.jpg output.jpg
def generateNewFace(filename, type, maskname):
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
    img = transforms.CenterCrop(img_size)(img)
    x = transforms.ToTensor()(img)
    x = torch.unsqueeze(x, dim=0)
    if 'png' in type:
        x = x[:, 0:3, :, :]

    # create mask
    # msk = generateMask(x.shape, img_w, img_h, int(x1), int(y1), int(x2), int(y2))
    msk = generateMask(x.shape, 160, 160, maskname)

    # inpaint
    with torch.no_grad():
        input = x - x * msk + mpv * msk
        output = model(input)
        inpainted = poisson_blend(input, output, msk)
        imgs = torch.cat((x, input, inpainted), dim=-1)
        imgs = save_image(imgs, output_img, nrow=3)
    print('output img was saved as %s.' % output_img)
    return output_filename
