import os
import numpy as np
from pytorch_lightning.utilities.cloud_io import load
import torch
import torch.nn as nn
from torch.utils import data
from unet_lightning import LitUNet as Unet
from dataset import AgVisionDataSet
from torch.utils.data import DataLoader

from torch.nn.functional import normalize, softmax
from torchvision.io import read_image

from PIL import Image

sft_mx = nn.Softmax2d()

mapping = {'bg': [0, 0, 0, 200],
            'dp': [255, 0, 0, 200],
            'dd': [0, 255, 0, 200],
            'er': [0, 0, 255, 200],
            'nd': [255, 255, 0, 200],
            'ps': [255, 0, 255, 200],
            'wa': [0, 255, 255, 200],
            'ww': [128, 128, 128, 200],
            'wc': [255, 255, 255, 200]
}

rgb_path = f"dataset/test/images/rgb/"


model = Unet.load_from_checkpoint('checkpoints/epoch=21-step=78297.ckpt')
model.eval()

test_data = AgVisionDataSet('test')
loader = iter(DataLoader(dataset=test_data, batch_size=1))

for i in range(len(os.listdir(rgb_path))):
    x, y, b, id = next(loader)

    label_path = 'labels/' + id[0] + '.png'

    print(id[0])

    z = model(x)
    z = sft_mx(z)
    z = torch.argmax(z, dim=1)
    z = z * b
    z = z.squeeze()
    z = z.numpy()
    z = z.astype(np.uint8)

    img = Image.fromarray(z)
    img.save(label_path)

    # z = np.array(Image.open(label_path))

    # seg_img = np.zeros((512,512,4))
    # seg_img[z == 1] = mapping['dp']
    # seg_img[z == 2] = mapping['dd']
    # seg_img[z == 3] = mapping['er']
    # seg_img[z == 4] = mapping['nd']
    # seg_img[z == 5] = mapping['ps']
    # seg_img[z == 6] = mapping['wa']
    # seg_img[z == 7] = mapping['ww']
    # seg_img[z == 8] = mapping['wc']

    # seg_img = seg_img.astype(np.uint8)

    # # # img = Image.fromarray(seg_img, 'RGB')
    # # # img.show()

    # background = Image.open("dataset/test/images/rgb/" + id[0] + ".jpg")
    # foreground = Image.fromarray(seg_img, 'RGBA')
    # # foreground.convert('RGBA')

    # background.paste(foreground, (0, 0), foreground)
    # background.show()