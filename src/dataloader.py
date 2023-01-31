import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from os.path import join
import random



class PatchDataset(Dataset):
    def __init__(self, image_path, image_ind, patch_size, dataset_size, bayer):

        self.patch_size = patch_size
        self.dataset_size = dataset_size
        self.image_ind = image_ind
        self.image_path = image_path
        self.bayer = bayer

        x = np.linspace(-1, 1, patch_size)
        y = np.linspace(-1, 1, patch_size)
        self.coords_x, self.coords_y = np.float32(np.meshgrid(x, y))

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_num = random.choice(self.image_ind)
        imageName = join(self.image_path, str(img_num).zfill(4)) + '.png'
        img_hr = np.flip(np.array(Image.open(imageName)),2)

        rand_y = random.randrange(img_hr.shape[0]-self.patch_size)
        rand_x = random.randrange(img_hr.shape[1]-self.patch_size)
        output_patch = img_hr[rand_y:(rand_y + self.patch_size), rand_x:(rand_x + self.patch_size),:]

        if np.max(output_patch) == 0 and np.min(output_patch) == 0:
            output_patch = np.ones_like(output_patch) * -1
        elif (np.max(output_patch)-np.min(output_patch)) == 0:
            output_patch = np.ones_like(output_patch) * 1
        else:
            output_patch = ((output_patch - np.min(output_patch)) / (
                        np.max(output_patch) - np.min(output_patch)) - 0.5) / 0.5

        if self.bayer == 'grbg':
            input_patch = bgr2GRBG(output_patch)
        else:
            raise ValueError(f'{self.bayer} bayer scheme is not available!!!')

        input_patch_LR4 = grbg2LR4(input_patch)

        input_coords_x = torch.tensor(self.coords_x.reshape(-1,1), dtype=torch.float32)
        input_coords_y = torch.tensor(self.coords_y.reshape(-1,1), dtype=torch.float32)
        input_patch = torch.tensor(input_patch, dtype=torch.float32)
        input_patch_LR4 = torch.tensor(input_patch_LR4, dtype=torch.float32)
        output_patch = torch.tensor(output_patch, dtype=torch.float32)

        return torch.stack((input_coords_y, input_coords_x), dim=-1).squeeze(1), input_patch, input_patch_LR4, output_patch

def bgr2GRBG(im_bgr):
    # (B,G,R) = cv.split(im_rgb)
    B = im_bgr[:,:,0]
    G = im_bgr[:,:,1]
    R = im_bgr[:,:,2]

    im_bayer = np.empty((im_bgr.shape[0], im_bgr.shape[1]), np.float32)
    im_bayer[0::2, 0::2] = G[0::2, 0::2] # top left
    im_bayer[0::2, 1::2] = R[0::2, 1::2] # top right
    im_bayer[1::2, 0::2] = B[1::2, 0::2] # bottom left
    im_bayer[1::2, 1::2] = G[1::2, 1::2] # bottom right

    return im_bayer

def grbg2LR4(im_bayer):
    LR4 =  np.empty([4,im_bayer.shape[0]//2, im_bayer.shape[1]//2], np.float32)
    LR4[0,:,:] = im_bayer[0::2, 0::2]
    LR4[1,:,:] = im_bayer[0::2, 1::2]
    LR4[2,:,:] = im_bayer[1::2, 0::2]
    LR4[3,:,:] = im_bayer[1::2, 1::2]

    return LR4

