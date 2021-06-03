import sys
import os
from PIL import Image
import numpy as np

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

id = 'XNFN16P34_2333-8641-2845-9153'
seg_path = f'/Users/zswanson/git-repos/agcompvis/labels/{id}.png'
img_path = f'/Users/zswanson/git-repos/agcompvis/dataset/test/images/rgb/{id}.jpg'

z = np.array(Image.open(seg_path))

seg_img = np.zeros((512,512,4))
seg_img[z == 1] = mapping['dp']
seg_img[z == 2] = mapping['dd']
seg_img[z == 3] = mapping['er']
seg_img[z == 4] = mapping['nd']
seg_img[z == 5] = mapping['ps']
seg_img[z == 6] = mapping['wa']
seg_img[z == 7] = mapping['ww']
seg_img[z == 8] = mapping['wc']

seg_img = seg_img.astype(np.uint8)

background = Image.open(img_path)
background.show()
foreground = Image.fromarray(seg_img, 'RGBA')

background.paste(foreground, (0, 0), foreground)
background.show()


# python test.py 8306 2900
# python test.py 11204 2900
# python test.py 14102 2900
# python test.py 17000 0