import sys
import os
from PIL import Image
import numpy as np

mapping = {'bg': [0, 0, 0, 200],         # background          = black
            'dp': [255, 0, 0, 200],      # double-plant        = red
            'dd': [0, 255, 0, 200],      # dry-down            = green
            'er': [0, 0, 255, 200],      # end row             = blue
            'nd': [255, 255, 0, 200],    # nutrient deficiency = yellow
            'ps': [255, 0, 255, 200],    # planter skip        = magenta
            'wa': [0, 255, 255, 200],    # water               = cyan
            'ww': [128, 128, 128, 200],  # waterway            = grey
            'wc': [255, 255, 255, 200]   # weed cluster        = white
}

# id = 'XNFN16P34_2333-8641-2845-9153'
# id = '1E3FJWUF1_7857-935-8369-1447'
# id = '1E3FJWUF1_8284-1120-8796-1632'
# id = '1FY8MBG8K_3050-1017-3562-1529'
# id = 'ZT13DBZBJ_3672-616-4184-1128'
# id = 'Y4GCKG3C6_5343-12261-5855-12773'
# id = 'Y4GCKG3C6_5321-4977-5833-5489'
# id = 'XWURP6UYI_7657-2259-8169-2771'
# id = 'WF332BJ2T_578-2309-1090-2821'
# id = 'VH13TNDT8_3772-3829-4284-4341'
# id = 'TU4111V1D_6672-3582-7184-4094'
# id = 'Q7UTPATN4_2594-3049-3106-3561'
# id = 'NQZ3Q6BA2_1766-1095-2278-1607'
# id = 'N12PKNH6L_15530-13042-16042-13554'
id = 'MG31WIR41_2763-5594-3275-6106'
# id = ''
# id = ''
# id = ''
# id = ''
# id = ''
# id = ''
# id = ''
# id = ''
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