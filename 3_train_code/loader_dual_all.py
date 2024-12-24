import numpy as np
import torch, torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from helper import load_tiff_stack_with_metadata, save_to_tiff_stack_with_metadata
from pathlib import Path

# from recon_fan import recon_fan_proj

import random

def normalize_image(image, MIN_B=-1024.0, MAX_B=3072.0):
   image = (image - MIN_B) / (MAX_B - MIN_B)
   image[image > 1] = 1
   image[image < 0] = 0
   return image

class ldct_dataset(Dataset):
    def __init__(self, ld_dose='12.5mAs', hd_dose='200mAs', mode='train', patch_n=None, patch_size=None, 
                 with_mask_sample=False, enable_feature_filter=False, patch_to_pixel_training=False):
        
        self.patch_n = patch_n
        self.patch_size = patch_size
        
        self.with_mask_sample = with_mask_sample
        self.patch_to_pixel_training = patch_to_pixel_training
        self.enable_feature_filter = enable_feature_filter
        
        self.mode = mode
        
        load_pa_list = ['L067', 'L096', 'L109', 'L143', 'L192', 'L286', 'L291', 'L310', 'L333', 'L506']
        
        self.ld_sino_list = []
        self.hd_sino_list = []
        self.meta_list = []
        for pa in load_pa_list:
            self.ld_sino_data, metas_pa = load_tiff_stack_with_metadata(Path(f'../{pa}_flat_fan_projections_{ld_dose}.tif'))
            self.hd_sino_data, _ = load_tiff_stack_with_metadata(Path(f'../{pa}_flat_fan_projections_{hd_dose}.tif'))
            self.ld_sino_data = (self.ld_sino_data[:,:, 30:80]/0.7).transpose(2, 0, 1)
            self.hd_sino_data = (self.hd_sino_data[:,:, 30:80]/0.7).transpose(2, 0, 1)
            self.ld_sino_data = np.copy(np.flip(self.ld_sino_data, axis=2))
            self.hd_sino_data = np.copy(np.flip(self.hd_sino_data, axis=2))
            self.ld_sino_data = np.expand_dims(self.ld_sino_data, axis=1)
            self.hd_sino_data = np.expand_dims(self.hd_sino_data, axis=1)
            self.ld_sino_list.append(self.ld_sino_data)
            self.hd_sino_list.append(self.hd_sino_data)
            self.meta_list.append(metas_pa)
        
        # for i in range(10):
        #     print(self.ld_sino_list[i].shape)
        self.ld_sino_data = np.concatenate(self.ld_sino_list, axis=0)
        self.hd_sino_data = np.concatenate(self.hd_sino_list, axis=0)      
        del self.ld_sino_list, self.hd_sino_list
        
        self.data_len_all = self.ld_sino_data.shape[0]            
        random.seed(0)
        index_numbers = list(range(self.data_len_all))
        random.shuffle(index_numbers)
        if self.mode == 'train':
            self.index_numbers = index_numbers[:450]
        else:
            self.index_numbers = index_numbers[450:]
            self.index_numbers = sorted(self.index_numbers)

    def __len__(self):
        return len(self.index_numbers)

    def __getitem__(self, idx):
        idx = self.index_numbers[idx]
        ld_sino = self.ld_sino_data[idx]
        hd_sino = self.hd_sino_data[idx]
        if self.patch_size:
            input_patches, target_patches = get_patch(ld_sino,
                                                        hd_sino,
                                                        self.patch_n,
                                                        self.patch_size,
                                                        2304, 736,
                                                        with_mask_sample=self.with_mask_sample,
                                                        patch_to_pixel_training=self.patch_to_pixel_training)
            return (input_patches, target_patches, idx)
        
        else:
            return (ld_sino, hd_sino, idx)
    
def get_patch(full_input_img, full_target_img, patch_n, patch_size, h, w, 
              with_mask_sample=False, patch_to_pixel_training=False):
    if with_mask_sample:
        assert patch_size % 2 == 1, "patch_size should be odd number under mask sampling"
        mask = (full_input_img[0, :, :] > 0.05).astype(full_input_img.dtype)
        mask[:patch_size//2] = 0
        mask[-patch_size//2+1:] = 0
        mask[:,:patch_size//2] = 0
        mask[:,-patch_size//2+1:] = 0
        mask = mask/np.sum(mask)
        indices = np.random.choice(h * w, size=patch_n, p=mask.flatten())
        top = (indices // w) - patch_size // 2
        left = (indices % w) - patch_size // 2
    else:    
        top = np.random.randint(0, h-patch_size, size=patch_n)
        left = np.random.randint(0, w-patch_size, size=patch_n)

    if patch_to_pixel_training:
        assert patch_size % 2 == 1, "patch_size should be odd number when doing patch_to_pixel training"
        patch_input_img = []
        for _i in range(patch_n):
            patch_input_img.append(full_input_img[:, top[_i]:top[_i]+patch_size, left[_i]:left[_i]+patch_size])
        patch_input_img = np.array(patch_input_img)
        patch_target_img = full_target_img[:, top + patch_size//2, left + patch_size//2]
    else:
        patch_input_img = []
        patch_target_img = []
        for _i in range(patch_n):
            patch_input_img.append(full_input_img[:, top[_i]:top[_i]+patch_size, left[_i]:left[_i]+patch_size])
            patch_target_img.append(full_target_img[:, top[_i]:top[_i]+patch_size, left[_i]:left[_i]+patch_size])
        patch_input_img = np.array(patch_input_img)
        patch_target_img = np.array(patch_target_img)
    return (patch_input_img, patch_target_img)


def data_loader_(ld_dose='12.5mAs', hd_dose='200mAs', mode='train', patch_n=None, patch_size=None, 
                 with_mask_sample=False, enable_feature_filter=False, patch_to_pixel_training=False, train_batch_size=1):
    if mode == 'train':
        data_loader_ = DataLoader(dataset=ldct_dataset(ld_dose, hd_dose, mode, patch_n, patch_size, with_mask_sample, enable_feature_filter, patch_to_pixel_training), 
                                  batch_size=train_batch_size, shuffle=False, num_workers=7)
    elif mode == 'valid':
        data_loader_ = DataLoader(dataset=ldct_dataset(ld_dose, hd_dose, mode, patch_n, patch_size, with_mask_sample, enable_feature_filter, patch_to_pixel_training), 
                                  batch_size=1, shuffle=False, num_workers=7)
    elif mode == 'all':
        data_loader_ = ldct_dataset(ld_dose, hd_dose, mode, patch_n, patch_size, with_mask_sample, enable_feature_filter, patch_to_pixel_training).get_ori()
    return data_loader_

if __name__ == '__main__':
    sample_patch_n = 10
    sample_patch_size = 65

    # train_loader = data_loader_(ld_dose='50mAs', hd_dose='200mAs', mode='train', patch_n=None, patch_size=None, 
    #                             with_mask_sample=False, enable_feature_filter=False, patch_to_pixel_training=False)
    # train_len = len(train_loader)

    valid_loader = data_loader_(ld_dose='50mAs', hd_dose='200mAs', mode='valid', patch_n=None, patch_size=None, 
                                with_mask_sample=False, enable_feature_filter=False, patch_to_pixel_training=False)
    valid_len = len(valid_loader)

    for i, (input_sino, target_sino, idx) in enumerate(valid_loader):
        print(input_sino.shape)
        print(target_sino.shape)
        print(i)
        print(idx)
        # x = input_img.cuda().view(-1, 1, sample_patch_size, sample_patch_size).float()
        # y = target_img.cuda().view(-1, 1, sample_patch_size, sample_patch_size).float()

        # x_patch = input_img.cuda().view(-1, 1, sample_patch_size, sample_patch_size).float()
        # y_pixel = target_img.cuda().view(-1, 1).float()