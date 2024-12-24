# Dual-Domain-CT-Denoising
A framework for dual domain CT denoising

This is a dual domain CT denoising pipeline, dataset from AAPM2016
input: low-dose projection data
output: reconstructed CT image

[rebin_code]
- for one2one reconsturction from single projection  
> noise_simulation.py: config the parser and run main.py for simulating low-dose projection  
> helper.py : [official code from helix2fan] load .tiff files  
> read_data.py: [official code from helix2fan] load .dicom files  
> rebinning_functions.py: [official code from helix2fan] for rebining *Change the parser config corresponding with the chart to get the abdominal slice region  

[model_code]
- the neuron network model in each domain  
> get_feature.py: Deterministic Filtering in image domain net, can work both inside and outside the network  
> net_img.py: model for image domain net (Patch2Pixel)  
> net_recon.py: model for differentiable FBP  
> net_sino.py: model for projection domain net (Patch2Pixel)  

[training_code]
- train the model  
> loader_dual_all.py: load the projection data  
> loader_s2.py: load the reconstructed image data  
> train_stage1.py: for projection domain training  
> train_stage2.py: for image domain training  

[metrics_code]
- image quality evaluation  
> cal_ssim.py: call for SSIM loss training; call for PSNR and SSIM calculating in measure2.py  
> measure2.py: Use compute_measure_simple to calculating PSNR, SSIM, RMSE at the same time  
> metrics.py: printProgressBar and save fig  
