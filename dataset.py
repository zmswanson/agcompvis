import torch
from torch import tensor
from torch.nn.functional import normalize
from torchvision import transforms
import torchvision
from torchvision.io import read_image
from torchvision.transforms import ToTensor
from torch.utils import data
import numpy as np
import os
from PIL import Image

class AgVisionDataSet(data.Dataset):
    def __init__(self, select_dataset="train", transform=None, start_index=0):
        if select_dataset in ['train', 'val', 'test']:
            self.dataset = select_dataset
        else:
            raise ValueError(f"{select_dataset} is not a valid dataset. Valid datasets include \"train\", \"val\", and \"test\".")

        rgb_path = f"dataset/{self.dataset}/images/rgb/"
        nir_path = f"dataset/{self.dataset}/images/nir/"

        self.rgb_inputs = [rgb_path + s for s in sorted(os.listdir(rgb_path))[start_index:]]
        self.nir_inputs = [nir_path + s for s in sorted(os.listdir(nir_path))[start_index:]]

        if self.dataset in ["train", "val"]:
            dp1_path = f"dataset/{self.dataset}/labels/double_plant/"
            dd2_path = f"dataset/{self.dataset}/labels/drydown/"
            er3_path = f"dataset/{self.dataset}/labels/endrow/"
            nd4_path = f"dataset/{self.dataset}/labels/nutrient_deficiency/"
            ps5_path = f"dataset/{self.dataset}/labels/planter_skip/"
            wa6_path = f"dataset/{self.dataset}/labels/water/"
            ww7_path = f"dataset/{self.dataset}/labels/waterway/"
            wc8_path = f"dataset/{self.dataset}/labels/weed_cluster/"

            self.double_plant = [dp1_path + s for s in sorted(os.listdir(dp1_path))[start_index:]]
            self.drydown      = [dd2_path + s for s in sorted(os.listdir(dd2_path))[start_index:]]
            self.endrow       = [er3_path + s for s in sorted(os.listdir(er3_path))[start_index:]]
            self.nutrient_def = [nd4_path + s for s in sorted(os.listdir(nd4_path))[start_index:]]
            self.planter_skip = [ps5_path + s for s in sorted(os.listdir(ps5_path))[start_index:]]
            self.water        = [wa6_path + s for s in sorted(os.listdir(wa6_path))[start_index:]]
            self.waterway     = [ww7_path + s for s in sorted(os.listdir(ww7_path))[start_index:]]
            self.weed_cluster = [wc8_path + s for s in sorted(os.listdir(wc8_path))[start_index:]]

        bound_path = f"dataset/{self.dataset}/boundaries/"
        self.boundaries = [bound_path + s for s in sorted(os.listdir(bound_path))[start_index:]]

        self.transform = transform
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.float32

    def __len__(self):
        return len(self.rgb_inputs)

    def __getitem__(self, index: int):
        # Select the sample
        rgb_input_ID = self.rgb_inputs[index]
        nir_input_ID = self.nir_inputs[index]
        boundary_ID  = self.boundaries[index]
        
        if self.dataset in ["train", "val"]:
            double_plant_ID = self.double_plant[index]
            drydown_ID      = self.drydown[index]
            endrow_ID       = self.endrow[index]
            nutrient_def_ID = self.nutrient_def[index]
            planter_skip_ID = self.planter_skip[index]
            water_ID        = self.water[index]
            waterway_ID     = self.waterway[index]
            weed_cluster_ID = self.weed_cluster[index]

        # Load input
        x0 = read_image(rgb_input_ID).type(self.inputs_dtype)
        x1 = read_image(nir_input_ID).type(self.inputs_dtype)
        x = normalize(torch.cat([x0, x1], dim=0)) # combine the nir and rgb data

        y = torch.zeros((1, 512, 512))

        if self.dataset in ["train", "val"]:
            y = torch.cat([y, read_image(double_plant_ID).type(self.targets_dtype) / 255], dim=0)
            y = torch.cat([y, read_image(drydown_ID).type(self.targets_dtype) * 2 / 255], dim=0)
            y = torch.cat([y, read_image(endrow_ID).type(self.targets_dtype) * 3 / 255], dim=0)
            y = torch.cat([y, read_image(nutrient_def_ID).type(self.targets_dtype) * 4 / 255], dim=0)
            y = torch.cat([y, read_image(planter_skip_ID).type(self.targets_dtype) * 5 / 255], dim=0)
            y = torch.cat([y, read_image(water_ID).type(self.targets_dtype) * 6 / 255], dim=0)
            y = torch.cat([y, read_image(waterway_ID).type(self.targets_dtype) * 7 / 255], dim=0)
            y = torch.cat([y, read_image(weed_cluster_ID).type(self.targets_dtype) * 8 / 255], dim=0)
            y = torch.argmax(y, dim=0, keepdim=True)

            y = torch.squeeze(y)

        # Preprocessing
        if self.transform is not None:
            x, y = self.transform(x, y)

        
        b = read_image(boundary_ID).type(self.inputs_dtype) / 255.0
        b = b[0, :, :]

        if self.dataset in ['test']:
            id = os.path.basename(rgb_input_ID).split('.')[0]
            return x, y, b, id

        b = torch.reshape(b, (1, 512, 512))
        b = torch.repeat_interleave(b, 9, dim=0)
        return x, y, b

if __name__ == '__main__':
    training_dataset = AgVisionDataSet(select_dataset='train', transform=None)

    training_dataloader = data.DataLoader(dataset=training_dataset, batch_size=2, shuffle=True)

    x, y = next(iter(training_dataloader))

    print(f'x = shape: {x.shape}; type: {x.dtype}')
    print(f'x = min: {x.min()}; max: {x.max()}')
    print(f'y = shape: {y.shape}; class: {y.unique()}; type: {y.dtype}')

    mapping = {'bg': (0, 0, 0),
               'dp': (255, 0, 0),
               'dd': (255, 128, 0),
               'er': (255, 255, 0),
               'nd': (128, 255, 0),
               'ps': (0, 255, 0),
               'wa': (255, 0, 255),
               'ww': (0, 0, 255),
               'wc': (128, 0, 255)
    }

    print(y[1])
    print(y[1].shape)
    z = torch.squeeze(y[1])

    seg_img = np.zeros((512,512,3))
    seg_img[z == 1] = mapping['dp']
    seg_img[z == 2] = mapping['dd']
    seg_img[z == 3] = mapping['er']
    seg_img[z == 4] = mapping['nd']
    seg_img[z == 5] = mapping['ps']
    seg_img[z == 6] = mapping['wa']
    seg_img[z == 7] = mapping['ww']
    seg_img[z == 8] = mapping['wc']

    img = Image.fromarray(seg_img, 'RGB')
    img.show()