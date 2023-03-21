import argparse
import math
import os,sys
sys.path.append(".") 
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# # os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from PIL import Image

import torch
import torchvision
from torch import optim
from tqdm import tqdm

from stylegan2.model import Generator,Discriminator
import clip
import numpy as np
import time
from smoothing import smooth, bilateralFilter
import torch.nn as nn
import cv2
from rembg.bg import get_mask,alpha_matting_cutout
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.nn import functional as F
import lpips
from get_masks import g_nonsaturating_loss
from pymatting import estimate_alpha_cf,estimate_foreground_ml,blend
from miscc.config import cfg,cfg_from_file

if cfg.CONFIG_NAME=='':
    cfg_from_file('code/cfg/eval_bird.yml')
stylegan_ckpt=cfg.STYLEGAN

discriminator=Discriminator(256,1)
discriminator.load_state_dict(torch.load(stylegan_ckpt)["d"])
discriminator=discriminator.cuda()
discriminator.eval()

def _resize_pil_image(image, size, resample="bicubic"):
    filters = {
        "bicubic": Image.BICUBIC,
        "bilinear": Image.BILINEAR,
        "box": Image.BOX,
        "hamming": Image.HAMMING,
        "lanczos": Image.LANCZOS,
        "nearest": Image.NEAREST,
        "none": Image.NEAREST,
    }

    size = (int(image.width * size), int(image.height * size))
    image = image.resize(size, filters[resample.lower()])

    return image

def get_lr(t, initial_lr, rampdown=0.2, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


def prepare(img_path,mask_path,ckpt_path,dir):
    tran = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    image=tran(Image.open(img_path).convert("RGB")).float().unsqueeze(0)
    image=nn.Upsample((256,256))(image)
    image=image.cuda()

    image_smoothed=smooth(img_path)
    image_smoothed=cv2.resize(image_smoothed,(256,256))
    path=os.path.join(dir,"smoothed.jpg")
    cv2.imwrite(path,image_smoothed)
    image_smoothed = tran(Image.open(path).convert("RGB")).float().unsqueeze(0)
    image_smoothed = image_smoothed.cuda()

    mask=cv2.imread(mask_path,0)
    ret, binary = cv2.threshold(mask,90,255,cv2.THRESH_BINARY)
    mask=transforms.ToTensor()(binary).float().unsqueeze(0).cuda()
    torchvision.utils.save_image(mask, os.path.join(dir,"mask.png"), normalize=True, range=(0, 1))

    g_ema = Generator(256, 512, 2,channel_multiplier=1)
    g_ema.load_state_dict(torch.load(ckpt_path)["g_ema"], strict=False)
    g_ema.eval()
    g_ema = g_ema.cuda()

    return image,image_smoothed,mask,g_ema

def latent_noise(latent, strength):
    noise = torch.randn_like(latent) * strength

    return latent + noise

def noise_regularize(noises):
    loss = 0

    for noise in noises:
        size = noise.shape[2]

        while True:
            loss = (
                loss
                + (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2)
                + (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2)
            )

            if size <= 8:
                break

            noise = noise.reshape([-1, 1, size // 2, 2, size // 2, 2])
            noise = noise.mean([3, 5])
            size //= 2

    return loss

def noise_normalize_(noises):
    for noise in noises:
        mean = noise.mean()
        std = noise.std()

        noise.data.add_(-mean).div_(std)

def make_image(tensor):
    return (
        tensor.detach()
        .clamp_(min=-1, max=1)
        .add(1)
        .div_(2)
        .mul(255)
        .type(torch.uint8)
        .permute(0, 2, 3, 1)
        .to("cpu")
        .numpy()
    )


def get_r(left,right,i,speed=2):
    r=(i-left)/(right-left)*speed
    r=1 if r>1 else r
    return r



def project(orig_img,imgs,masks,img_target,g_ema,percept,dir):
    _step=1200
    _step0=300
    _step1=500
    _step2=800
    _step3=1000
    _lr=0.1
    _a_lmd=3
    _g_lmd=0.08
    batch_size=1

    n_mean_latent = 10000

    with torch.no_grad():
        noise_sample = torch.randn(n_mean_latent, 512).cuda()
        latent_out = g_ema.style(noise_sample)

        latent_mean = latent_out.mean(0)
        latent_std = ((latent_out - latent_mean).pow(2).sum() / n_mean_latent) ** 0.5

    latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(batch_size, 1)
    latent_in.requires_grad = True
    # latent_in2=torch.randn(batch_size,512).cuda()
    # latent_in2=g_ema.style(latent_in2).detach()
    # latent_in2.requires_grad = True

    optimizer = optim.Adam([latent_in],lr=_lr)

    pbar = tqdm(range(_step))
    latent_path = []
    mask0=get_mask(orig_img.unsqueeze(0))
    # mask00=get_mask(imgs[0].unsqueeze(0))
    # a=torch.sum(mask0[mask00>mask0])/256/256
    # if a.item()>0.08:
    #     mask0=mask00
    #     print('change shape')
    # else:
    #     mask0[mask00>mask0]=mask00[mask00>mask0]
    p_loss = percept(mask0, mask0).sum()

    parts=[]
    for i in range(len(imgs)):
        masks[i]=masks[i]*mask0.squeeze(0)
        parts.append(imgs[i]*masks[i].unsqueeze(0))
    parts=torch.cat(parts)

    img_target=img_target*mask0+orig_img.unsqueeze(0)*(1-mask0)

    # area=np.ones(len(masks))
    # for i in range(len(masks)):
    #     area[i]=torch.sum(masks[i]>0.01)
    # area=1/area*area.max()

    # print(area)

    torchvision.utils.save_image(orig_img, os.path.join(dir,"target0.png"), normalize=True, range=(-1, 1))
    torchvision.utils.save_image(img_target, os.path.join(dir,"target.png"), normalize=True, range=(-1, 1))
    torchvision.utils.save_image(mask0, os.path.join(dir,"shape.png"), normalize=True, range=(0, 1))
    torchvision.utils.save_image(masks, os.path.join(dir,"masks.png"), nrow=4, normalize=True, range=(0, 1))
    torchvision.utils.save_image(parts, os.path.join(dir,"parts.png"), nrow=4, normalize=True, range=(-1, 1))
    # assert(0)

    # target=mask0*img_target
    target=mask0*orig_img
    g_loss=torch.zeros(1).cuda()
    for i in pbar:
        t = i / _step
        lr = get_lr(t, _lr)
        optimizer.param_groups[0]["lr"] = lr

        img_gen, _ = g_ema([latent_in], input_is_latent=True, randomize_noise=False)

        # mask1=get_mask(img_gen)
        # mask1=mask1.detach()
        if i == _step0:
            target=mask0*img_target

        # if i == _step2:
        #     # target=img_target
        #     _a_lmd*=3

        if i == _step3:
            target=img_gen.clone().detach() 

        if i >= _step1 and i < _step2:
            r=get_r(_step1,_step2,i)
            a_lmd=_a_lmd*r  
        elif i >= _step2 and i < _step3:
            r=get_r(_step2,_step3,i)
            # cur_mask=mask0*(1-r)+r
            # target=img_target*cur_mask
            a_lmd=_a_lmd+_a_lmd*r
        elif i>=_step3:
            r=get_r(_step3,_step,i,4)
            g_lmd=_g_lmd*r          

        shape_loss=percept(img_gen,target)    
        loss = shape_loss

        if i > _step1 and i < _step3:
            for j in range(len(masks)):
                img_gen_part=img_gen*masks[j]
                img_target_part=imgs[j]*masks[j]
                p_loss = percept(img_gen_part, img_target_part).sum()
                loss+=a_lmd*p_loss

        if i>=_step3:
            fake_pred = discriminator(img_gen)
            g_loss = g_nonsaturating_loss(fake_pred)
            loss+=g_lmd*g_loss

        pbar.set_description(
            (
                f"shape: {shape_loss.item():.4f}; attention: {p_loss.item():.4f}; real: {g_loss.item():.4f}; lr: {lr:.4f}"
            )
        )
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            latent_path.append(latent_in.detach().clone())
            torchvision.utils.save_image(img_gen, os.path.join(dir,"{}.png".format(i)), normalize=True, range=(-1, 1))

    mask1=get_mask(img_gen)

    filename = os.path.join(dir,'result.pt')
    img_ar = make_image(img_gen)

    result_file = {
        "img": img_gen[0],
        "latent": latent_in,
    }

    mask2=mask1.mul(255).type(torch.uint8).to("cpu").numpy()[0,0]
    mask2[mask2<10]=0
    mask2[mask2>240]=255
    m1=10<=mask2
    m2=mask2<=240
    grey=m1&m2
    mask2[grey]=127

    mask0[mask0<0.5]=0
    _mask=mask0.mul(255).type(torch.uint8).to("cpu").numpy()[0,0]
    orig_img=make_image(orig_img.unsqueeze(0))[0]
    orig_img=cv2.cvtColor(orig_img,cv2.COLOR_RGB2BGR)
    background = cv2.inpaint(orig_img, _mask, 3,cv2.INPAINT_NS)
    background = cv2.inpaint(background, _mask, 3,cv2.INPAINT_NS)
    background=cv2.cvtColor(np.array(background),cv2.COLOR_BGR2RGB)

    Image.fromarray(background).save(os.path.join(dir,"backgroung.png"))

    background=_resize_pil_image(Image.fromarray(background), 1.0, "box")
    img_final=_resize_pil_image(Image.fromarray(img_ar[0]), 1.0, "box")
    mask2=_resize_pil_image(Image.fromarray(mask2), 1.0, "nearest")

    background = np.array(background) / 255.0
    img_final = np.array(img_final) / 255.0
    mask2 = np.array(mask2) / 255.0

    alpha = estimate_alpha_cf(img_final, mask2)
    foreground = estimate_foreground_ml(img_final, alpha)
    new_image = blend(foreground, background, alpha)

    new_image = np.clip(new_image * 255, 0, 255).astype(np.uint8)
    pil_img=Image.fromarray(new_image)

    img_name = os.path.join(dir,'project.png')
    # pil_img = Image.fromarray(background.numpy())
    pil_img.save(img_name)

    torch.save(result_file, filename)

    return pil_img, latent_in

def project_fast(orig_img,imgs,masks,img_target,g_ema,percept,dir):
    _step=1000
    _step0=300
    _step1=500
    _step2=800
    _step3=1000
    _lr=0.1
    _a_lmd=3
    _g_lmd=0.08
    batch_size=1

    n_mean_latent = 10000

    with torch.no_grad():
        noise_sample = torch.randn(n_mean_latent, 512).cuda()
        latent_out = g_ema.style(noise_sample)

        latent_mean = latent_out.mean(0)
        latent_std = ((latent_out - latent_mean).pow(2).sum() / n_mean_latent) ** 0.5

    latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(batch_size, 1)
    latent_in.requires_grad = True
    # latent_in2=torch.randn(batch_size,512).cuda()
    # latent_in2=g_ema.style(latent_in2).detach()
    # latent_in2.requires_grad = True

    optimizer = optim.Adam([latent_in],lr=_lr)

    pbar = tqdm(range(_step))
    latent_path = []
    mask0=get_mask(orig_img)
    # mask00=get_mask(imgs[0].unsqueeze(0))
    # a=torch.sum(mask0[mask00>mask0])/256/256
    # if a.item()>0.08:
    #     mask0=mask00
    #     print('change shape')
    # else:
    #     mask0[mask00>mask0]=mask00[mask00>mask0]
    p_loss = percept(mask0, mask0).sum()

    parts=[]
    for i in range(len(imgs)):
        masks[i]=masks[i]*mask0.squeeze(0)
        parts.append(imgs[i]*masks[i].unsqueeze(0))
    parts=torch.cat(parts)

    # area=np.ones(len(masks))
    # for i in range(len(masks)):
    #     area[i]=torch.sum(masks[i]>0.01)
    # area=1/area*area.max()

    # print(area)

    torchvision.utils.save_image(orig_img, os.path.join(dir,"target0.png"), normalize=True, range=(-1, 1))
    torchvision.utils.save_image(img_target, os.path.join(dir,"target.png"), normalize=True, range=(-1, 1))
    torchvision.utils.save_image(mask0, os.path.join(dir,"shape.png"), normalize=True, range=(0, 1))
    torchvision.utils.save_image(masks, os.path.join(dir,"masks.png"), nrow=4, normalize=True, range=(0, 1))
    torchvision.utils.save_image(parts, os.path.join(dir,"parts.png"), nrow=4, normalize=True, range=(-1, 1))
    # assert(0)

    # target=mask0*img_target
    target=mask0*orig_img
    g_loss=torch.zeros(1).cuda()
    for i in pbar:
        t = i / _step
        lr = get_lr(t, _lr)
        optimizer.param_groups[0]["lr"] = lr

        img_gen, _ = g_ema([latent_in], input_is_latent=True, randomize_noise=False)

        # if i == _step2:
        #     # target=img_target
        #     _a_lmd*=3

        if i == _step3:
            target=img_gen.clone().detach() 

        if i >= _step1 and i < _step2:
            r=get_r(_step1,_step2,i)
            a_lmd=_a_lmd*r  
        elif i >= _step2 and i < _step3:
            r=get_r(_step2,_step3,i)
            # cur_mask=mask0*(1-r)+r
            # target=img_target*cur_mask
            a_lmd=_a_lmd+_a_lmd*r
        elif i>=_step3:
            r=get_r(_step3,_step,i,4)
            g_lmd=_g_lmd*r          

        shape_loss=percept(img_gen,target)    
        loss = shape_loss

        if i > _step1 and i < _step3:
            for j in range(len(masks)):
                img_gen_part=img_gen*masks[j]
                img_target_part=orig_img*masks[j]
                p_loss = percept(img_gen_part, img_target_part).sum()
                loss+=a_lmd*p_loss

        if i>=_step3:
            fake_pred = discriminator(img_gen)
            g_loss = g_nonsaturating_loss(fake_pred)
            loss+=g_lmd*g_loss

        pbar.set_description(
            (
                f"shape: {shape_loss.item():.4f}; attention: {p_loss.item():.4f}; real: {g_loss.item():.4f}; lr: {lr:.4f}"
            )
        )
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            latent_path.append(latent_in.detach().clone())
            torchvision.utils.save_image(img_gen, os.path.join(dir,"{}.png".format(i)), normalize=True, range=(-1, 1))

    mask1=get_mask(img_gen)

    filename = os.path.join(dir,'result.pt')
    img_ar = make_image(img_gen)

    result_file = {
        "img": img_gen[0],
        "latent": latent_in,
    }

    mask2=mask1.mul(255).type(torch.uint8).to("cpu").numpy()[0,0]
    mask2[mask2<10]=0
    mask2[mask2>240]=255
    m1=10<=mask2
    m2=mask2<=240
    grey=m1&m2
    mask2[grey]=127

    mask0[mask0<0.5]=0
    _mask=mask0.mul(255).type(torch.uint8).to("cpu").numpy()[0,0]
    orig_img=make_image(orig_img)[0]
    orig_img=cv2.cvtColor(orig_img,cv2.COLOR_RGB2BGR)
    background = cv2.inpaint(orig_img, _mask, 3,cv2.INPAINT_NS)
    background = cv2.inpaint(background, _mask, 3,cv2.INPAINT_NS)
    background=cv2.cvtColor(np.array(background),cv2.COLOR_BGR2RGB)

    Image.fromarray(background).save(os.path.join(dir,"backgroung.png"))

    background=_resize_pil_image(Image.fromarray(background), 1.0, "box")
    img_final=_resize_pil_image(Image.fromarray(img_ar[0]), 1.0, "box")
    mask2=_resize_pil_image(Image.fromarray(mask2), 1.0, "nearest")

    background = np.array(background) / 255.0
    img_final = np.array(img_final) / 255.0
    mask2 = np.array(mask2) / 255.0

    alpha = estimate_alpha_cf(img_final, mask2)
    foreground = estimate_foreground_ml(img_final, alpha)
    new_image = blend(foreground, background, alpha)

    new_image = np.clip(new_image * 255, 0, 255).astype(np.uint8)
    pil_img=Image.fromarray(new_image)

    img_name = os.path.join(dir,'project.png')
    # pil_img = Image.fromarray(background.numpy())
    pil_img.save(img_name)

    torch.save(result_file, filename)

    return pil_img, latent_in


if __name__ == "__main__":
    p='output/features/output0.png'
    mask_p='output/features/attention0.png'

    ckpt='pretrained_models/bird_stylegan.pt'
    dir='output/projector'
    image_target,image_smoothed,mask,g_ema=prepare(p,mask_p,ckpt,dir)
    project(image_target,image_smoothed,mask,g_ema,dir)
    print()

