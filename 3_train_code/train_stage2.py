import numpy as np
import torch, torch.nn as nn
import torch.optim as optim
import os
import time
import random
from torch.utils.data import Dataset, DataLoader
from glob import glob

from measure2 import compute_measure_simple
from metrics import printProgressBar, save_fig

# set trunc and denorm
trunc_min = -1024.0
trunc_max = 3072.0
show_trunc_min = -160.0
show_trunc_max = 240.0
def trunc_img(image):
    image[image < trunc_min] = trunc_min
    image[image > trunc_max] = trunc_max
    return image
def normalize_img(image, MIN_Image=-1024.0, MAX_Image=3072.0):
    image = (image - MIN_Image) / (MAX_Image - MIN_Image)
    image[image > 1] = 1
    image[image < 0] = 0
    return image
def denormalize_img(image, MIN_Image=-1024.0, MAX_Image=3072.0):
    image = image * (MAX_Image - MIN_Image) + MIN_Image
    return image
def trunc_sino(mat):
    mat[mat <= 0.0] = 0.0
    mat[mat >= 1.0] = 1.0
    return mat

### save and load
def save_model(i):
    if not os.path.exists(saved_model_path):
        os.makedirs(saved_model_path)
        print('Create path : {}'.format(saved_model_path))

    f = os.path.join(saved_model_path, '{}epoch.ckpt'.format(i))
    torch.save(model.state_dict(), f)

def load_model(model_, model_path, iter_):
    f = os.path.join(model_path, '{}epoch.ckpt'.format(iter_))
    model_.load_state_dict(torch.load(f))

### seed
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True  

# loader
load_dose = '50mAs'
from loader_s2 import data_loader_
train_loader = data_loader_(ld_dose=load_dose, hd_dose='200mAs', mode='train', patch_n=10, patch_size=65, 
                            with_mask_sample=True, enable_feature_filter=True, patch_to_pixel_training=False, train_batch_size=5)
train_len = len(train_loader)

valid_loader = data_loader_(ld_dose=load_dose, hd_dose='200mAs', mode='valid', patch_n=None, patch_size=None,
                            with_mask_sample=False, enable_feature_filter=True, patch_to_pixel_training=False)
valid_len = len(valid_loader)

# hyperparameters
num_epochs = 5001
count = 0
best_iters = 0
patch_size = 5
total_iters = train_len
print_iters = total_iters//5
print_ep = 5
train_losses = []
val_losses = []

psnr_npy = np.zeros(1)
ssim_npy = np.zeros(1)
psnr_sino_npy = np.zeros(1)
ssim_sino_npy = np.zeros(1)

# model
from net_img import Estimate_Noise2_FC, Estimate_Noise2_Infer
model_img = Estimate_Noise2_FC(9, 64).cuda()
model = Estimate_Noise2_Infer(model_img)

print('Parameters:', sum(param.numel() for param in model.parameters()))

### optimizer
from losses_ssim import SSIMLoss
criterion_ssim = SSIMLoss().cuda()
criterion_mse = nn.MSELoss().cuda()
learning_rate = 0.00001
optimizer = optim.Adam(model.parameters(), lr=learning_rate) 

# save model
save_epoch = 20 # 5-10%ep, one save
val_epoch = 20 # 0.5-1%ep, one valid
saved_model_path = f'../model_50mAs_s2_p2p_ssim'

# for load model to continue training
continue_epoch = -1
load_model_path = f'../model_50mAs_s2_p2p_ssim'
if continue_epoch != -1:
    load_model(model, load_model_path, continue_epoch)

## train
start_time = time.time()
for epoch in range(num_epochs):
    setup_seed(20)

    for i, (ld_img, filtered_img, hd_img, f_name, idx) in enumerate(train_loader):
        count += 1

        x = filtered_img.view(-1, 9, 65, 65).cuda().float()
        y = hd_img.view(-1, 1, 65, 65).cuda().float()

        # print(input_sino.shape, target_sino.shape)
        
        model.zero_grad()
        optimizer.zero_grad()

        pred = model(x)        
        loss = criterion_ssim(pred, y)

        loss.backward()
        train_losses.append(loss.item())
        optimizer.step()

    if epoch % print_ep == 0:                        
            print("STEP [{}], EPOCH [{}/{}], ITER [{}/{}] \nLOSS: {:.16f}, TIME: {:.1f}s".format(count, epoch, 
                                                                                                num_epochs, i+1, 
                                                                                                total_iters, loss.item(), 
                                                                                                time.time() - start_time))

    if epoch % save_epoch == 0:
        save_model(epoch+continue_epoch+1)
        np.save(os.path.join(saved_model_path, 'loss_{}_epoch.npy'.format(epoch+continue_epoch+1)), np.array(train_losses))
        print('Model saved at {} epoch'.format(epoch+continue_epoch+1))


    if epoch % val_epoch == 0:
        # compute PSNR, SSIM, RMSE
        with torch.no_grad():
            pred_result_all = torch.zeros(2)
            for val_i, (val_ld_img, filtered_img, val_hd_img, f_name, val_idx) in enumerate(valid_loader):
                
                # model_infer = Estimate_Noise2_Infer(model)

                pred = model(filtered_img.cuda().float())
                
                x = trunc_img(denormalize_img(val_ld_img[:, 0].view(512,512).cpu().float()))
                y = trunc_img(denormalize_img(val_hd_img.view(512,512).cpu().float()))
                pred = trunc_img(denormalize_img(pred.view(512,512).cpu().float()))
                
                pred_result = compute_measure_simple(x, y, pred, trunc_min, trunc_max)
                pred_result = torch.from_numpy(pred_result).squeeze()
                pred_result_all += pred_result
                printProgressBar(val_i, valid_len, prefix="Compute measurements ..", suffix='Complete', length=25)

            pred_result_avg = pred_result_all/valid_len

            print('\n')
            print("pred_psnr:", f"{pred_result_avg[0].item():.4f}")
            print("pred_ssim:", f"{pred_result_avg[1].item():.4f}")

            psnr_npy = np.concatenate((psnr_npy, np.array([pred_result_avg[0].item()])), axis=0)
            ssim_npy = np.concatenate((ssim_npy, np.array([pred_result_avg[1].item()])), axis=0)

            np.save(f'{saved_model_path}/psnr.npy', psnr_npy)
            np.save(f'{saved_model_path}/ssim.npy', ssim_npy)

