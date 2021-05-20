# %%
import os
from numpy import AxisError, average
from numpy.lib.type_check import imag

train_path = '/Users/zswanson/Downloads/supervised/Agriculture-Vision-2021/train/'
rgb_path   = train_path + 'images/rgb/'
label_path = train_path + 'labels/'
mask_path  = train_path + 'masks/'
bound_path = train_path + 'boundaries/'

labels = ['double_plant',
          'drydown',
          'endrow',
          'nutrient_deficiency',
          'planter_skip',
          'storm_damage',
          'water',
          'waterway',
          'weed_cluster']
# %%
from skimage.io import imread
import numpy as np
# %%
with open(train_path + 'train.csv', 'w') as csv:
    csv.write("file," +
              "field," +
              "x1,y1,x2,y2," +
              "double_plant," +
              "drydown," +
              "endrow," +
              "nutrient_deficiency," +
              "planter_skip," +
              "storm_damage," +
              "water," +
              "waterway," +
              "weed_cluster," +
              "mask,"
              "boundary\n")

    for img in os.listdir(rgb_path):
        csv.write(img + ',')
        csv.write(img.split('_')[0] + ',')
        csv.write(img.split('_')[1].split('-')[0] + ',')
        csv.write(img.split('_')[1].split('-')[1] + ',')
        csv.write(img.split('_')[1].split('-')[2] + ',')
        csv.write(img.split('_')[1].split('-')[3].split('.')[0] + ',')

        for l in labels:
            if np.average(imread(label_path + l + '/' + img.split('.')[0] + '.png')) > 0.0:
                csv.write('1.0,')
            else:
                csv.write('0.0,')

        if np.average(imread(mask_path + img.split('.')[0] + '.png')) < 255.0:
            csv.write('1.0,')
        else:
            csv.write('0.0,')

        if np.average(imread(bound_path + img.split('.')[0] + '.png')) < 255.0:
            csv.write('1.0\n')
        else:
            csv.write('0.0\n')
# %%
import pandas as pd

train_meta = pd.read_csv(train_path + 'train.csv')

# %%
train_meta.info()
# %%
import matplotlib.pyplot as plt

ax = train_meta.plot.hist(by=labels)
# %%
image_count = {}

image_count['mask'] = train_meta['mask'].value_counts()[1.0]
print("[*] Mask images:",image_count['mask'])

image_count['boundary'] = train_meta['boundary'].value_counts()[1.0]
print("[*] Boundary images:",image_count['boundary'])

for l in labels:
    image_count[l] = train_meta[l].value_counts()[1.0]
    print("[*]",l.capitalize(),"images:", image_count[l])

image_count = dict(sorted(image_count.items(), key=lambda item: item[1], reverse=True))
# %%
plt.bar(range(len(image_count)), list(image_count.values()), align='center')
plt.xticks(range(len(image_count)), list(image_count.keys()), rotation=90)

plt.show()