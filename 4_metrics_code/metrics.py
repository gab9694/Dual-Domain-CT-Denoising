import torch
import numpy as np
# from piq import psnr, ssim, FID, iw_ssim, fsim, gmsd, tv, vsiGAB_LDCT/LEI/RED-CNN/measure.py

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# def compute_metrics_simple(x, y, pred, trunc_min, trunc_max):


#     x_bchw = x.unsqueeze(0).unsqueeze(0) - trunc_min
#     y_bchw = y.unsqueeze(0).unsqueeze(0) - trunc_min
#     pred_bchw = pred.unsqueeze(0).unsqueeze(0) - trunc_min
#     data_range = trunc_max - trunc_min
    
#     original_psnr = psnr(x_bchw, y_bchw, data_range=data_range)
#     original_ssim = ssim(x_bchw, y_bchw, data_range=data_range)

#     pred_psnr = psnr(pred_bchw, y_bchw, data_range=data_range)
#     pred_ssim = ssim(pred_bchw, y_bchw, data_range=data_range)

#     original_result = []
#     pred_result = []
#     original_result.append((original_psnr, original_ssim))
#     pred_result.append((pred_psnr, pred_ssim))
#     original_result = torch.stack([_ for _ in original_result[0]])
#     pred_result = torch.stack([_ for _ in pred_result[0]])

#     return original_result, pred_result

# def compute_metrics(x, y, pred, trunc_min, trunc_max):


#     x_bchw = x.unsqueeze(0).unsqueeze(0) - trunc_min
#     y_bchw = y.unsqueeze(0).unsqueeze(0) - trunc_min
#     pred_bchw = pred.unsqueeze(0).unsqueeze(0) - trunc_min
#     data_range = trunc_max - trunc_min
    
#     fid_metric = FID()

#     x_3channel = torch.zeros((1, 3, 512, 512))
#     x_3channel[:, 1, :, :] = x_bchw
#     y_3channel = torch.zeros((1, 3, 512, 512))
#     y_3channel[:, 1, :, :] = y_bchw
#     pred_3channel = torch.zeros((1, 3, 512, 512))
#     pred_3channel[:, 1, :, :] = pred_bchw

#     x_tv = tv.total_variation(x_bchw, norm_type='l2', reduction='mean')
#     y_tv = tv.total_variation(y_bchw, norm_type='l2', reduction='mean')
#     pred_tv = tv.total_variation(pred_bchw, norm_type='l2', reduction='mean')

#     original_psnr = psnr(x_bchw, y_bchw, data_range=data_range)
#     original_ssim = ssim(x_bchw, y_bchw, data_range=data_range)
#     original_iwssim = iw_ssim.information_weighted_ssim(x_bchw, y_bchw, data_range=data_range)
#     original_fid = fid_metric(x, y)
#     original_gmsd = gmsd(x_bchw, y_bchw, data_range=data_range)
#     original_fsim = fsim(x_3channel, y_3channel, data_range=data_range)
#     original_vsi = vsi(x_3channel, y_3channel, reduction='mean', data_range=data_range)
#     original_tv_diff = abs(x_tv - y_tv)

#     pred_psnr = psnr(pred_bchw, y_bchw, data_range=data_range)
#     pred_ssim = ssim(pred_bchw, y_bchw, data_range=data_range)
#     pred_iwssim = iw_ssim.information_weighted_ssim(pred_bchw, y_bchw, data_range=data_range)
#     pred_fid = fid_metric(pred, y)
#     pred_gmsd = gmsd(pred_bchw, y_bchw, data_range=data_range)
#     pred_fsim = fsim(pred_3channel, y_3channel, data_range=data_range)
#     pred_vsi = vsi(pred_3channel, y_3channel, reduction='mean', data_range=data_range)
#     pred_tv_diff = abs(pred_tv - y_tv)

#     original_result = []
#     pred_result = []
#     original_result.append((original_psnr, original_ssim, original_iwssim, original_fsim, original_fid, original_gmsd, original_vsi, x_tv, y_tv, original_tv_diff))
#     pred_result.append((pred_psnr, pred_ssim, pred_iwssim,pred_fsim, pred_fid, pred_gmsd, pred_vsi, pred_tv, y_tv, pred_tv_diff))

#     original_result = torch.stack([_ for _ in original_result[0]])
#     pred_result = torch.stack([_ for _ in pred_result[0]])

#     return original_result, pred_result


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill=' '):
    # referred from https://gist.github.com/snakers4/91fa21b9dda9d055a02ecd23f24fbc3d
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '=' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    if iteration == total:
        print()

def save_fig_simple(x, y, pred, fig_save_path, fig_name, original_result, pred_result, trunc_min=-160, trunc_max=240):
    x, y, pred = x.numpy(), y.numpy(), pred.numpy()
    f, ax = plt.subplots(1, 3, figsize=(30, 10))
    ax[0].imshow(x, cmap=plt.cm.gray, vmin=trunc_min, vmax=trunc_max)
    ax[0].set_title('Quarter-dose', fontsize=30)
    ax[0].set_xlabel("PSNR: {:.4f}   SSIM: {:.4f}\nIWSSIM:{:.4f}   FSIM: {:.4f}\nGMSD: {:.4f}       VSI: {:.4f}".format(original_result[0].item(),
                                                                        original_result[1].item(), 0, 0, 0, 0,
                                                                        # original_result[2].item(),
                                                                        # original_result[3].item(),
                                                                        # original_result[5].item(),
                                                                        # original_result[6].item(),
                                                                        ), fontsize=20)
    ax[1].imshow(pred, cmap=plt.cm.gray, vmin=trunc_min, vmax=trunc_max)
    ax[1].set_title('Result', fontsize=30)
    ax[1].set_xlabel("PSNR: {:.4f}   SSIM: {:.4f}\nIWSSIM:{:.4f}   FSIM: {:.4f}\nGMSD: {:.4f}       VSI: {:.4f}".format(pred_result[0].item(),
                                                                    pred_result[1].item(), 0, 0, 0, 0,
                                                                    # pred_result[2].item(),
                                                                    # pred_result[3].item(),
                                                                    # pred_result[5].item(),
                                                                    # pred_result[6].item(),
                                                                    ), fontsize=20)
    ax[2].imshow(y, cmap=plt.cm.gray, vmin=trunc_min, vmax=trunc_max)
    ax[2].set_title('Full-dose', fontsize=30)

    f.savefig(os.path.join(fig_save_path, 'result_{}.png'.format(fig_name)))
    plt.close()

def save_fig(x, y, pred, fig_save_path, fig_name, original_result, pred_result, trunc_min=-160, trunc_max=240):
    x, y, pred = x.numpy(), y.numpy(), pred.numpy()
    f, ax = plt.subplots(1, 3, figsize=(30, 10))
    ax[0].imshow(x, cmap=plt.cm.gray, vmin=trunc_min, vmax=trunc_max)
    ax[0].set_title('Quarter-dose', fontsize=30)
    ax[0].set_xlabel("PSNR: {:.4f}   SSIM: {:.4f}\nIWSSIM:{:.4f}   FSIM: {:.4f}\nGMSD: {:.4f}       VSI: {:.4f}".format(original_result[0].item(),
                                                                        original_result[1].item(),
                                                                        original_result[2].item(),
                                                                        original_result[3].item(),
                                                                        original_result[5].item(),
                                                                        original_result[6].item(),
                                                                        ), fontsize=20)
    ax[1].imshow(pred, cmap=plt.cm.gray, vmin=trunc_min, vmax=trunc_max)
    ax[1].set_title('Result', fontsize=30)
    ax[1].set_xlabel("PSNR: {:.4f}   SSIM: {:.4f}\nIWSSIM:{:.4f}   FSIM: {:.4f}\nGMSD: {:.4f}       VSI: {:.4f}".format(pred_result[0].item(),
                                                                    pred_result[1].item(), 
                                                                    pred_result[2].item(),
                                                                    pred_result[3].item(),
                                                                    pred_result[5].item(),
                                                                    pred_result[6].item(),
                                                                    ), fontsize=20)
    ax[2].imshow(y, cmap=plt.cm.gray, vmin=trunc_min, vmax=trunc_max)
    ax[2].set_title('Full-dose', fontsize=30)

    f.savefig(os.path.join(fig_save_path, 'result_{}.png'.format(fig_name)))
    plt.close()
