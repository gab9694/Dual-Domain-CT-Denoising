import numpy as np
import torch, torch.nn as nn
import torch.optim as optim
import os
import time
import random
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # 使用 GPU 3

from loader_dual_all import data_loader_
from helper import load_tiff_stack_with_metadata, save_to_tiff_stack_with_metadata
from pathlib import Path

from measure2 import compute_measure_simple
from metrics import printProgressBar

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

dose_list = ['12.5mAs', '25mAs', '50mAs', '75mAs', '100mAs', '150mAs', '200mAs']
# for dose in dose_list:
#     print(f'Processing {dose} dose')
# sample_patch_n = 10
# sample_patch_size = 65
load_pa='L506'
load_dose = '50mAs'
train_loader = data_loader_(ld_dose=load_dose, hd_dose='200mAs', mode='train', patch_n=10000, patch_size=5, 
                            with_mask_sample=False, enable_feature_filter=False, patch_to_pixel_training=True, train_batch_size=1)
train_len = len(train_loader)

valid_loader = data_loader_(ld_dose=load_dose, hd_dose='200mAs', mode='valid', patch_n=None, patch_size=None, 
                            with_mask_sample=False, enable_feature_filter=False, patch_to_pixel_training=False)
valid_len = len(valid_loader)
_, metas = load_tiff_stack_with_metadata(Path(f'../{load_pa}_flat_fan_projections_200mAs.tif'))

load_pa_list = ['L067', 'L096', 'L109', 'L143', 'L192', 'L286', 'L291', 'L310', 'L333', 'L506']
metas_list = []
for load_pa in load_pa_list:
    _, metas = load_tiff_stack_with_metadata(Path(f'../{load_pa}_flat_fan_projections_200mAs.tif'))
    metas_list.append(metas)


# hyperparameters
num_epochs = 5001
count = 0
best_iters = 0
patch_size = 5
total_iters = train_len
print_iters = total_iters//5
train_losses = []
val_losses = []

psnr_npy = np.zeros(1)
ssim_npy = np.zeros(1)
psnr_sino_npy = np.zeros(1)
ssim_sino_npy = np.zeros(1)
# model
from model.net_sino import Estimate_Noise_FC, Estimate_Noise_Infer
model = Estimate_Noise_FC(1, 64).cuda()

from model.net_recon import Recon_Only_Net


print('Parameters:', sum(param.numel() for param in model.parameters()))

### optimizer
criterion_mse = nn.MSELoss().cuda()
learning_rate = 0.00001
optimizer = optim.Adam(model.parameters(), lr=learning_rate) 

# save model
save_epoch = 5 # 5-10%ep, one save
val_epoch = 5 # 0.5-1%ep, one valid
saved_model_path = f'../model_50mAs2'

# for load model to continue training
continue_epoch = -1
load_model_path = f'../model_50mAs2'
if continue_epoch != -1:
    load_model(model, load_model_path, continue_epoch)

### train
start_time = time.time()
for epoch in range(num_epochs):
    setup_seed(20)
    # for i in index_train_numbers:
    for i, (input_sino, target_sino, idx) in enumerate(train_loader):
        count += 1

        input_sino = input_sino.cuda().squeeze(0).float()
        target_sino = target_sino.cuda().view(-1,1).float()
        # print(input_sino.shape, target_sino.shape)
        
        model.zero_grad()
        optimizer.zero_grad()

        filtered_sinogram = model(input_sino)        
        loss = criterion_mse(filtered_sinogram.cuda().float(), target_sino.cuda().float())

        loss.backward()
        train_losses.append(loss.item())
        optimizer.step()

    if epoch % 5 == 0:                        
            print("STEP [{}], EPOCH [{}/{}], ITER [{}/{}] \nLOSS: {:.16f}, TIME: {:.1f}s".format(count, epoch, 
                                                                                                num_epochs, i+1, 
                                                                                                total_iters, loss.item(), 
                                                                                                time.time() - start_time))

    if epoch % 20 == 0:
        save_model(epoch+continue_epoch+1)
        np.save(os.path.join(saved_model_path, 'loss_{}_epoch.npy'.format(epoch+continue_epoch+1)), np.array(train_losses))
        print('Model saved at {} epoch'.format(epoch+continue_epoch+1))

    if epoch % 20 == 0:
        # compute PSNR, SSIM, RMSE
        with torch.no_grad():
            pred_result_all = torch.zeros(2)
            for val_i, (val_ld_sino, val_hd_sino, idx) in enumerate(valid_loader):
                
                model_recon = Recon_Only_Net(metas_list[idx//50]).cuda()
                
                model_infer = Estimate_Noise_Infer(model)
                filtered_sinogram = model_infer(val_ld_sino.cuda().float())        
                out_img = model_recon(filtered_sinogram.cuda().float())
                
                val_ld_img = model_recon(val_ld_sino.cuda().float())
                val_hd_img = model_recon(val_hd_sino.cuda().float())
                
                x = trunc_img(denormalize_img(val_ld_img.view(512,512).cpu().float()))
                y = trunc_img(denormalize_img(val_hd_img.view(512,512).cpu().float()))
                pred = trunc_img(denormalize_img(out_img.view(512,512).cpu().float()))
                
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
