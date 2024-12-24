import numpy as np
from torch.utils.data import Dataset, DataLoader
from glob import glob
import os

class ldct_dataset(Dataset):
    def __init__(self, ld_dose='12.5mAs', hd_dose='200mAs', mode='train', patch_n=None, patch_size=None, 
                 with_mask_sample=False, enable_feature_filter=False, patch_to_pixel_training=False):
        
        self.patch_n = patch_n
        self.patch_size = patch_size
        
        self.with_mask_sample = with_mask_sample
        self.patch_to_pixel_training = patch_to_pixel_training
        self.enable_feature_filter = enable_feature_filter
        if mode == 'train':
            data_path = '../stage2_data'
        elif mode == 'valid':
            data_path = '../stage2_data_test'

        if self.enable_feature_filter:
            ori_path = sorted(glob(os.path.join(data_path,  'F/*_original_imgF.npy')))
            filtered_path = sorted(glob(os.path.join(data_path,  'F/*_filtered_imgF.npy')))
        else:
            ori_path = sorted(glob(os.path.join(data_path,  '*_original_img.npy')))
            filtered_path = sorted(glob(os.path.join(data_path,  '*_filtered_img.npy')))
        target_path = sorted(glob(os.path.join(data_path,  '*_target_img.npy')))

        self.file_names = [_[-18:-15] for _ in target_path]
        if self.enable_feature_filter:
            self.ori_data = [np.load(_) for _ in ori_path]
            self.filtered_data = [np.load(_) for _ in filtered_path]
        else:
            self.ori_data = [np.expand_dims(np.load(_), axis=0) for _ in ori_path]
            self.filtered_data = [np.expand_dims(np.load(_), axis=0) for _ in filtered_path]
        self.target_data = [np.expand_dims(np.load(_), axis=0) for _ in target_path]

        if self.patch_size:
            for i in range(len(self.target_data)):
                ld_img = self.ori_data[i]
                hd_img = self.target_data[i]
                filtered_img = self.filtered_data[i]
                self.ori_data[i], self.filtered_data[i], self.target_data[i] = get_patch(ld_img,
                                                                            filtered_img,
                                                                            hd_img,
                                                                            self.patch_n,
                                                                            self.patch_size,
                                                                            512, 512,
                                                                            with_mask_sample=self.with_mask_sample,
                                                                            patch_to_pixel_training=self.patch_to_pixel_training)        
  
    def __len__(self):
        return len(self.target_data)

    def __getitem__(self, idx):
        ld_img = self.ori_data[idx]
        hd_img = self.target_data[idx]
        filtered_img = self.filtered_data[idx]
        file_name = self.file_names[idx]
        return (ld_img, filtered_img, hd_img, file_name, idx)
    
def get_patch(full_input_img, full_filter_img, full_target_img, patch_n, patch_size, h, w, 
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
        patch_filter_img = []
        for _i in range(patch_n):
            patch_input_img.append(full_input_img[:, top[_i]:top[_i]+patch_size, left[_i]:left[_i]+patch_size])
            patch_filter_img.append(full_filter_img[:, top[_i]:top[_i]+patch_size, left[_i]:left[_i]+patch_size])
        patch_input_img = np.array(patch_input_img)
        patch_filter_img = np.array(patch_filter_img)
        patch_target_img = full_target_img[:, top + patch_size//2, left + patch_size//2]
    else:
        patch_input_img = []
        patch_filter_img = []
        patch_target_img = []
        for _i in range(patch_n):
            patch_input_img.append(full_input_img[:, top[_i]:top[_i]+patch_size, left[_i]:left[_i]+patch_size])
            patch_filter_img.append(full_filter_img[:, top[_i]:top[_i]+patch_size, left[_i]:left[_i]+patch_size])
            patch_target_img.append(full_target_img[:, top[_i]:top[_i]+patch_size, left[_i]:left[_i]+patch_size])
        patch_input_img = np.array(patch_input_img)
        patch_filter_img = np.array(patch_filter_img)
        patch_target_img = np.array(patch_target_img)
    return (patch_input_img, patch_filter_img, patch_target_img)


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
    # train_loader = data_loader_(ld_dose='50mAs', hd_dose='200mAs', mode='train', patch_n=4000, patch_size=5, 
    #                             with_mask_sample=True, enable_feature_filter=True, patch_to_pixel_training=True, train_batch_size=5)
    # print(len(train_loader))

    # for i, (ld_img, filtered_img, hd_img, f_name, idx) in enumerate(train_loader):
    #     print(ld_img.shape, filtered_img.shape, hd_img.shape, f_name, idx)
    #     break

    train_loader = data_loader_(ld_dose='50mAs', hd_dose='200mAs', mode='valid', patch_n=None, patch_size=None, 
                                with_mask_sample=True, enable_feature_filter=True, patch_to_pixel_training=True, train_batch_size=1)
    print(len(train_loader))

    for i, (ld_img, filtered_img, hd_img, f_name, idx) in enumerate(train_loader):
        print(ld_img.shape, filtered_img.shape, hd_img.shape, f_name, idx)
        break