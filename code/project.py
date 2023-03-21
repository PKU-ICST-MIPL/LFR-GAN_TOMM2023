# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Project given image to the latent space of pretrained network pickle."""
import sys
sys.path.append(".") 
import copy
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
from time import perf_counter

import click
import imageio
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import clip
from rembg.bg import get_mask
import torchvision
from get_masks import g_nonsaturating_loss
import cv2
from miscc.config import cfg,cfg_from_file
import dnnlib
import legacy
from stylegan2.model import Discriminator
from tqdm import tqdm

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

def project(
    G,
    target: torch.Tensor, # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
    *,
    num_steps                  = 1000,
    w_avg_samples              = 10000,
    initial_learning_rate      = 0.1,
    initial_noise_factor       = 0.05,
    lr_rampdown_length         = 0.25,
    lr_rampup_length           = 0.05,
    noise_ramp_length          = 0.75,
    regularize_noise_weight    = 1e5,
    verbose                    = False,
    device: torch.device
):
    assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)

    def logprint(*args):
        if verbose:
            print(*args)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device) # type: ignore

    # Compute w stats.
    logprint(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None)  # [N, L, C]
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)       # [N, 1, C]
    w_avg = np.mean(w_samples, axis=0, keepdims=True)      # [1, 1, C]
    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

    # Setup noise inputs.
    noise_bufs = { name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name }

    # Load VGG16 feature detector.
    # url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    url = 'pretrained_models/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    # Features for target image.
    target_images = target.unsqueeze(0).to(device).to(torch.float32)
    if target_images.shape[2] > 256:
        target_images = F.interpolate(target_images, size=(256, 256), mode='area')
    target_features = vgg16(target_images, resize_images=False, return_lpips=True)

    w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=True) # pylint: disable=not-callable
    w_out = torch.zeros([num_steps] + list(w_opt.shape[1:]), dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)

    with torch.no_grad():
        clip_model, _ = clip.load("ViT-B/32", device=device)
        clip_model = clip_model.eval()
        txt = 'this bird has wings that are green and grey and has a red belly and blue head'
        tokenized_txt = clip.tokenize([txt]).to(device)
        txt_fts = clip_model.encode_text(tokenized_txt)
        txt_fts = txt_fts/txt_fts.norm(dim=-1, keepdim=True)

    # Init noise.
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    for step in range(num_steps):
        # Learning rate schedule.
        t = step / num_steps
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Synth images from opt_w.
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = (w_opt + w_noise).repeat([1, G.mapping.num_ws, 1])
        synth_images = G.synthesis(ws, fts=txt_fts, noise_mode='const')

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images = (synth_images + 1) * (255/2)
        if synth_images.shape[2] > 256:
            synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

        # Features for synth images.
        synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
        dist = (target_features - synth_features).square().sum()

        # Noise regularization.
        reg_loss = 0.0
        for v in noise_bufs.values():
            noise = v[None,None,:,:] # must be [1,1,H,W] for F.avg_pool2d()
            while True:
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=3)).mean()**2
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=2)).mean()**2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)
        loss = dist + reg_loss * regularize_noise_weight

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        logprint(f'step {step+1:>4d}/{num_steps}: dist {dist:<4.2f} loss {float(loss):<5.2f}')

        # Save projected W for each optimization step.
        w_out[step] = w_opt.detach()[0]

        # Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

    return w_out.repeat([1, G.mapping.num_ws, 1]), txt_fts

def get_r(left,right,i,speed=2):
    r=(i-left)/(right-left)*speed
    r=1 if r>1 else r
    return r

def project2(orig_img,imgs,masks,img_target,g_ema,percept,dir,w_opt,txt_fts,orig):
    _step=110
    _step0=30
    _step1=50
    _step2=80
    _step3=100
    _a_lmd=3
    _g_lmd=0.08

    w_avg_samples              = 10000
    initial_learning_rate      = 2e-7
    initial_noise_factor       = 0.05
    lr_rampdown_length         = 0.25
    lr_rampup_length           = 0.05
    noise_ramp_length          = 0.75
    regularize_noise_weight    = 1e5
    verbose                    = False

    device = orig_img.device
    G = copy.deepcopy(g_ema).eval().requires_grad_(False).to(device) # type: ignore

    # Compute w stats.
    # z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    # w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None)  # [N, L, C]
    # w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)       # [N, 1, C]
    # w_avg = np.mean(w_samples, axis=0, keepdims=True)      # [1, 1, C]
    # w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

    # Setup noise inputs.
    noise_bufs = { name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name }

    # Load VGG16 feature detector.
    # url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    url = 'pretrained_models/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    # Features for target image.
    mask0=get_mask(orig_img.unsqueeze(0))
    target_images = mask0*orig_img
    if target_images.shape[2] > 256:
        target_images = F.interpolate(target_images, size=(256, 256), mode='area')
    target_features = vgg16(target_images, resize_images=False, return_lpips=True)

    # w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=True) # pylint: disable=not-callable
    # w_out = torch.zeros([_step] + list(w_opt.shape[1:]), dtype=torch.float32, device=device)
    # optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)
    optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)

    # Init noise.
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True


    parts=[]
    for i in range(len(imgs)):
        masks[i]=masks[i]*mask0.squeeze(0)
        parts.append(imgs[i]*masks[i].unsqueeze(0))
    parts=torch.cat(parts)

    img_target=img_target*mask0+orig_img.unsqueeze(0)*(1-mask0)

    # torchvision.utils.save_image(orig_img, os.path.join(dir,"target0.png"), normalize=True, range=(-1, 1))
    # torchvision.utils.save_image(img_target, os.path.join(dir,"target.png"), normalize=True, range=(-1, 1))
    # torchvision.utils.save_image(mask0, os.path.join(dir,"shape.png"), normalize=True, range=(0, 1))
    # torchvision.utils.save_image(masks, os.path.join(dir,"masks.png"), nrow=4, normalize=True, range=(0, 1))
    # torchvision.utils.save_image(parts, os.path.join(dir,"parts.png"), nrow=4, normalize=True, range=(-1, 1))
    # torchvision.utils.save_image(orig, os.path.join(dir,"orig.png"), normalize=True, range=(-1, 1))
    # assert(0)

    g_loss=torch.zeros(1).cuda()
    p_loss=torch.zeros(1).cuda()

    for step in tqdm(range(_step)):
        # Learning rate schedule.
        t = step / _step
        # w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Synth images from opt_w.
        # w_noise = torch.randn_like(w_opt) * w_noise_scale
        # ws = (w_opt + w_noise).repeat([1, G.mapping.num_ws, 1])
        ws = (w_opt).repeat([1, G.mapping.num_ws, 1])
        synth_images = G.synthesis(ws, fts=txt_fts, noise_mode='const')
        if i == _step0:
            target_images=mask0*img_target
            target_features = vgg16(target_images, resize_images=False, return_lpips=True)
        # if i == _step2:
        #     # target=img_target
        #     _a_lmd*=3

        if i == _step3:
            target_images=synth_images.clone().detach()
            target_features = vgg16(target_images, resize_images=False, return_lpips=True) 

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

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images = (synth_images + 1) * (255/2)
        if synth_images.shape[2] > 256:
            synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

        # Features for synth images.
        synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
        dist = (target_features - synth_features).square().sum()

        # Noise regularization.
        reg_loss = 0.0
        for v in noise_bufs.values():
            noise = v[None,None,:,:] # must be [1,1,H,W] for F.avg_pool2d()
            while True:
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=3)).mean()**2
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=2)).mean()**2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)
        loss = dist + reg_loss * regularize_noise_weight

        if i > _step1 and i < _step3:
            for j in range(len(masks)):
                img_gen_part=synth_images*masks[j]
                img_target_part=imgs[j]*masks[j]
                p_loss = percept(img_gen_part, img_target_part).sum()
                loss+=a_lmd*p_loss

        if i>=_step3:
            fake_pred = discriminator(synth_images)
            g_loss = g_nonsaturating_loss(fake_pred)
            loss+=g_lmd*g_loss

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Save projected W for each optimization step.
        # w_out[step] = w_opt.detach()[0]

        # Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

    mask1=get_mask(synth_images)

    filename = os.path.join(dir,'result.pt')
    img_ar = make_image(synth_images)

    result_file = {
        "img": synth_images[0],
        "latent": ws.detach().cpu(),
    }

    # mask2=mask1.mul(255).type(torch.uint8).to("cpu").numpy()[0,0]
    # mask2[mask2<10]=0
    # mask2[mask2>240]=255
    # m1=10<=mask2
    # m2=mask2<=240
    # grey=m1&m2
    # mask2[grey]=127

    # mask0[mask0<0.5]=0
    # _mask=mask0.mul(255).type(torch.uint8).to("cpu").numpy()[0,0]
    # orig_img=make_image(orig_img.unsqueeze(0))[0]
    # orig_img=cv2.cvtColor(orig_img,cv2.COLOR_RGB2BGR)
    # background = cv2.inpaint(orig_img, _mask, 3,cv2.INPAINT_NS)
    # background = cv2.inpaint(background, _mask, 3,cv2.INPAINT_NS)
    # background=cv2.cvtColor(np.array(background),cv2.COLOR_BGR2RGB)

    # Image.fromarray(background).save(os.path.join(dir,"backgroung.png"))

    # background=_resize_pil_image(Image.fromarray(background), 1.0, "box")
    # img_final=_resize_pil_image(Image.fromarray(img_ar[0]), 1.0, "box")
    # mask2=_resize_pil_image(Image.fromarray(mask2), 1.0, "nearest")

    # background = np.array(background) / 255.0
    # img_final = np.array(img_final) / 255.0
    # mask2 = np.array(mask2) / 255.0

    # alpha = estimate_alpha_cf(img_final, mask2)
    # foreground = estimate_foreground_ml(img_final, alpha)
    # new_image = blend(foreground, background, alpha)

    # new_image = np.clip(new_image * 255, 0, 255).astype(np.uint8)
    # pil_img=Image.fromarray(new_image)

    # img_name = os.path.join(dir,'project.png')
    # # pil_img = Image.fromarray(background.numpy())
    # pil_img.save(img_name)

    # torch.save(result_file, filename)

    return Image.fromarray(img_ar[0]), ws.detach()



#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--target', 'target_fname', help='Target image file to project to', required=True, metavar='FILE')
@click.option('--num-steps',              help='Number of optimization steps', type=int, default=1000, show_default=True)
@click.option('--seed',                   help='Random seed', type=int, default=303, show_default=True)
@click.option('--save-video',             help='Save an mp4 video of optimization progress', type=bool, default=True, show_default=True)
@click.option('--outdir',                 help='Where to save the output images', required=True, metavar='DIR')
def run_projection(
    network_pkl: str,
    target_fname: str,
    outdir: str,
    save_video: bool,
    seed: int,
    num_steps: int
):
    """Project given image to the latent space of pretrained network pickle.

    Examples:

    \b
    python projector.py --outdir=out --target=~/mytargetimg.png \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load networks.
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as fp:
        G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device) # type: ignore

    # Load target image.
    target_pil = Image.open(target_fname).convert('RGB')
    w, h = target_pil.size
    s = min(w, h)
    target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    target_pil = target_pil.resize((G.img_resolution, G.img_resolution), Image.LANCZOS)
    target_uint8 = np.array(target_pil, dtype=np.uint8)

    # Optimize projection.
    start_time = perf_counter()
    projected_w_steps, txt_fts = project(
        G,
        target=torch.tensor(target_uint8.transpose([2, 0, 1]), device=device), # pylint: disable=not-callable
        num_steps=num_steps,
        device=device,
        verbose=True
    )
    print (f'Elapsed: {(perf_counter()-start_time):.1f} s')

    # Render debug output: optional video and projected image and W vector.
    os.makedirs(outdir, exist_ok=True)
    if save_video:
        video = imageio.get_writer(f'{outdir}/proj.mp4', mode='I', fps=10, codec='libx264', bitrate='16M')
        print (f'Saving optimization progress video "{outdir}/proj.mp4"')
        for projected_w in projected_w_steps:
            synth_image = G.synthesis(projected_w.unsqueeze(0), fts=txt_fts, noise_mode='const')
            synth_image = (synth_image + 1) * (255/2)
            synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
            video.append_data(np.concatenate([target_uint8, synth_image], axis=1))
        video.close()

    # Save final projected frame and W vector.
    target_pil.save(f'{outdir}/target.png')
    projected_w = projected_w_steps[-1]
    synth_image = G.synthesis(projected_w.unsqueeze(0), fts=txt_fts, noise_mode='const')
    synth_image = (synth_image + 1) * (255/2)
    synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    Image.fromarray(synth_image, 'RGB').save(f'{outdir}/proj.png')
    np.savez(f'{outdir}/projected_w.npz', w=projected_w.unsqueeze(0).cpu().numpy())

#----------------------------------------------------------------------------

if __name__ == "__main__":
    
    run_projection() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------