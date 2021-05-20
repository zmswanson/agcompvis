import torch
from torchvision.io import read_image
from torchvision.transforms import ToTensor
from torch.utils import data
import numpy as np


class AgVisionDataSet(data.Dataset):
    def __init__(self, rgb_inputs: list, nir_inputs: list, targets: list, transform=None):
        self.rgb_inputs = rgb_inputs
        self.nir_inputs = nir_inputs
        self.targets = targets
        self.transform = transform
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long

    def __len__(self):
        return len(self.rgb_inputs)

    def __getitem__(self, index: int):
        # Select the sample
        rgb_input_ID = self.rgb_inputs[index]
        nir_input_ID = self.nir_inputs[index]
        target_ID = self.targets[index]

        # Load input and target
        x, y, z = read_image(rgb_input_ID).type(self.inputs_dtype), read_image(target_ID).type(self.targets_dtype), read_image(nir_input_ID).type(self.inputs_dtype)
        x = torch.cat([x,z], dim=0) # combine the nir and rgb data

        # Preprocessing
        if self.transform is not None:
            x, y = self.transform(x, y)

        return x, y

rgb_inputs = ['/Users/zswanson/Downloads/supervised/Agriculture-Vision-2021/train/images/rgb/11IE4DKTR_11556-9586-12068-10098.jpg', '/Users/zswanson/Downloads/supervised/Agriculture-Vision-2021/train/images/rgb/11IE4DKTR_6121-684-6633-1196.jpg']
nir_inputs = ['/Users/zswanson/Downloads/supervised/Agriculture-Vision-2021/train/images/nir/11IE4DKTR_11556-9586-12068-10098.jpg', '/Users/zswanson/Downloads/supervised/Agriculture-Vision-2021/train/images/nir/11IE4DKTR_6121-684-6633-1196.jpg']
targets = ['/Users/zswanson/Downloads/supervised/Agriculture-Vision-2021/train/labels/double_plant/11IE4DKTR_11556-9586-12068-10098.png', '/Users/zswanson/Downloads/supervised/Agriculture-Vision-2021/train/labels/double_plant/11IE4DKTR_6121-684-6633-1196.png']

training_dataset = AgVisionDataSet(rgb_inputs=rgb_inputs,
                                       nir_inputs=nir_inputs,
                                       targets=targets,
                                       transform=ToTensor())

training_dataloader = data.DataLoader(dataset=training_dataset,
                                      batch_size=2,
                                      shuffle=True)
x, y = next(iter(training_dataloader))

print(f'x = shape: {x.shape}; type: {x.dtype}')
print(f'x = min: {x.min()}; max: {x.max()}')
print(f'y = shape: {y.shape}; class: {y.unique()}; type: {y.dtype}')