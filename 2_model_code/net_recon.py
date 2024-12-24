import torch, torch.nn as nn
import numpy as np
from torch_radon import RadonFanbeam

def normalize_img(image, MIN_Image=-1024.0, MAX_Image=3072.0):
    image = (image - MIN_Image) / (MAX_Image - MIN_Image)
    return image

class Recon_Only_Net(nn.Module):
    def __init__(self, metadata=None):
        super().__init__()

        '''
        metadata: correct info from dicom header
        forward input: single rebined sinogram data

        Here, RadonFanbeam work separately as single forward pass.
        In fact, this could combine into any forward pass of networks.
        '''

        self.angles = np.array(metadata['angles'])[:metadata['rotview']] + (np.pi / 2)
        self.vox_scaling = 1 / 0.7
        self.radon = RadonFanbeam(512,
                            self.angles,
                            source_distance=self.vox_scaling * metadata['dso'],
                            det_distance=self.vox_scaling * metadata['ddo'],
                            det_count=736,
                            det_spacing=self.vox_scaling * metadata['du'],
                            clip_to_circle=False) 

    def forward(self, x):
        filtered_sinogram = self.radon.filter_sinogram(x.squeeze(), filter_name='shepp-logan')
        fbp = self.radon.backprojection(filtered_sinogram)
        fbp_hu = normalize_img(1000 * ((fbp - 0.0192) / 0.0192)).unsqueeze(0).unsqueeze(0)
        return fbp_hu
    
x= torch.randn(1, 1, 512, 512).cuda()
input_sin = x.cuda().squeeze