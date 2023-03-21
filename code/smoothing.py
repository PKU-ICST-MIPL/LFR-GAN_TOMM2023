import cv2
from rembg.bg import remove, PIL_remove
import numpy as np
import io
from PIL import Image
from PIL import ImageFile
import torch
import torch.nn.functional as F  
ImageFile.LOAD_TRUNCATED_IMAGES = True


def smooth2(input_path,output_path):
    f = np.fromfile(input_path)
    result = remove(f)
    img = Image.open(io.BytesIO(result)).convert("RGBA")
    image = Image.new('RGB', size=img.size, color=(255, 255, 255))
    image.paste(img, (0, 0), mask=img)
    image = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR) 
    image=cv2.bilateralFilter(image,30,75,75)
    cv2.imwrite(output_path,image)

def smooth(img_path):
    f = np.fromfile(img_path)
    result = remove(f)
    img = Image.open(io.BytesIO(result)).convert("RGBA")
    image = Image.new('RGB', size=img.size, color=(255, 255, 255))
    image.paste(img, (0, 0), mask=img)
    image = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR) 
    image = cv2.bilateralFilter(image,30,75,75)
    return image

def smooth_PIL(img):
    mask, result = PIL_remove(img, return_mask=True)
    img = Image.open(io.BytesIO(result)).convert("RGBA")
    image = Image.new('RGB', size=img.size, color=(255, 255, 255))
    image.paste(img, (0, 0), mask=img)
    image = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR) 
    image = cv2.bilateralFilter(image,30,75,75)
    return mask, image

@torch.no_grad()
def getGaussianKernel(ksize, sigma=0):
    if sigma <= 0:
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8 
    center = ksize // 2
    xs = (np.arange(ksize, dtype=np.float32) - center) 
    kernel1d = np.exp(-(xs ** 2) / (2 * sigma ** 2)) 
    kernel = kernel1d[..., None] @ kernel1d[None, ...] 
    kernel = torch.from_numpy(kernel)
    kernel = kernel / kernel.sum() 
    return kernel

def GaussianBlur(batch_img, ksize, sigma=None):
    kernel = getGaussianKernel(ksize, sigma) 
    B, C, H, W = batch_img.shape 
    kernel = kernel.view(1, 1, ksize, ksize).repeat(C, 1, 1, 1)
    pad = (ksize - 1) // 2 
    batch_img_pad = F.pad(batch_img, pad=[pad, pad, pad, pad], mode='reflect')
    weighted_pix = F.conv2d(batch_img_pad, weight=kernel, bias=None, 
                           stride=1, padding=0, groups=C)
    return weighted_pix

def bilateralFilter(batch_img, ksize, sigmaColor=None, sigmaSpace=None):
    device = batch_img.device
    if sigmaSpace is None:
        sigmaSpace = 0.15 * ksize + 0.35  # 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    if sigmaColor is None:
        sigmaColor = sigmaSpace
    
    pad = (ksize - 1) // 2
    batch_img_pad = F.pad(batch_img, pad=[pad, pad, pad, pad], mode='reflect')
    
    # patches.shape:  B x C x H x W x ksize x ksize
    patches = batch_img_pad.unfold(2, ksize, 1).unfold(3, ksize, 1)
    patch_dim = patches.dim() # 6 
    diff_color = patches - batch_img.unsqueeze(-1).unsqueeze(-1)
    weights_color = torch.exp(-(diff_color ** 2) / (2 * sigmaColor ** 2))
    weights_color = weights_color / weights_color.sum(dim=(-1, -2), keepdim=True)
    
    weights_space = getGaussianKernel(ksize, sigmaSpace).to(device)
    weights_space_dim = (patch_dim - 2) * (1,) + (ksize, ksize)
    weights_space = weights_space.view(*weights_space_dim).expand_as(weights_color)
    
    weights = weights_space * weights_color
    weights_sum = weights.sum(dim=(-1, -2))
    weighted_pix = (weights * patches).sum(dim=(-1, -2)) / weights_sum
    return weighted_pix

if __name__=='__main__':
    p='datasets/birds/images/016.Painted_Bunting/Painted_Bunting_0091_15198.jpg'
    o='output/smoothed.png'
    smooth2(p,o)