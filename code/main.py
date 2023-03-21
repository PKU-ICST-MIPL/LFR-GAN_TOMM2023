import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

from sentence_parser import parser
import numpy as np
import torchvision.transforms as transforms
import torchvision
from PIL import Image
from stylegan2.model import Generator
import torch
from miscc.config import cfg,cfg_from_file
import pickle


from pymatting import *
import lpips
from model import RNN_ENCODER,G_DCGAN,G_NET
import traceback
# from nltk.tokenize import RegexpTokenizer
from get_masks import G_NET2,get_mask2, get_mask3
from projector5 import project,project_fast
from project import project2
from miscc.utils import get_code, get_sentences, load_dataset_dict_info
import dnnlib
import legacy
import clip
import argparse

args = argparse.ArgumentParser()
args.add_argument("-c", "--cfg", type=str, default='code/cfg/eval_bird.yml', help="cfg file path")
args.add_argument("-l", "--lafite", type=str, default='pretrained_models/birds.pkl',help="lafite checkpoint")
args.add_argument("-s", "--space", type=str, default='lafite',help="latent space used to modify image")
args = vars(args.parse_args())

cfg_from_file(args['cfg'])
stylegan_ckpt=(cfg.STYLEGAN)
tran = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

tran_64 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((64,64)),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
output_dir='output/'

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

def prepare_model():
    percept = lpips.PerceptualLoss(
        model="net-lin", net="vgg"
    )
    ps=parser()

    # ensure_checkpoint_exists(stylegan_ckpt)
    # g_ema = Generator(256, 512, 2,channel_multiplier=1)
    # g_ema.load_state_dict(torch.load(stylegan_ckpt)["g_ema"], strict=False)
    # g_ema.eval()
    # g_ema = g_ema.cuda()
    with dnnlib.util.open_url(args['lafite']) as fp:
        g_ema = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to('cuda') # type: ignore

    ixtoword, wordtoix,n_words=load_dataset_dict_info()
    # Build and load the generator
    text_encoder = RNN_ENCODER(n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
    state_dict = torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
    text_encoder.load_state_dict(state_dict)
    print('Load text encoder from:', cfg.TRAIN.NET_E)
    text_encoder = text_encoder.cuda()
    text_encoder.eval()

    model_dir = cfg.TRAIN.NET_G
    state_dict = \
        torch.load(model_dir, map_location=lambda storage, loc: storage)

    netG2 = G_NET2()
    netG2.load_state_dict(state_dict)
    netG2.cuda()
    netG2.eval()

    clip_model, _ = clip.load("ViT-B/32", device='cuda')
    clip_model = clip_model.eval()

    return percept,g_ema,text_encoder,netG2,ps,clip_model


def _main(sentences):
    percept,g_ema,text_encoder,netG2,ps,_=prepare_model()

    g_ema = Generator(256, 512, 2,channel_multiplier=1)
    g_ema.load_state_dict(torch.load(stylegan_ckpt)["g_ema"], strict=False)
    g_ema.eval()
    g_ema = g_ema.cuda()

    projector_dir=os.path.join(output_dir,"projector")
    attn_dir=os.path.join(output_dir,"features")

    parsed=ps.parse(sentences)

    # get feature images and attention images, match it to the origin image
    # parsed: {0: (-1, 'A bird'), 3: (1, 'a blue tail'), 7: (0, 'red feet'), ...}
    print(parsed)
    # assert(0)
    imgs,masks,orig_img=get_mask2(parsed,attn_dir,text_encoder,netG2)

    img_target=imgs[0].clone()
    n=len(masks)

    masks_join=torch.zeros((3,256,256)).cuda()
    remain_idx=[]
    for i in range(n-1,-1,-1):
        masks[i]=transforms.ToTensor()(masks[i]).cuda()
        masks[i][masks[i]<0.1]=0
        m1,m2=masks[i]>0.1, masks[i]>masks_join
        m3,m4=masks[i]>0.2, masks_join>0.2
        ratio1=torch.sum(m1 & m2)/(torch.sum(m1).float()+1e-3)
        ratio2=torch.sum(m3 & m4)/(torch.sum(m3).float()+1e-3)
        print(ratio1,ratio2)
        remain=ratio1>=0.4 and ratio2<=0.9
        remain_idx.append(remain)
        if not remain:
            continue
        masks_join[masks[i]>masks_join]=masks[i][masks[i]>masks_join]

    remain_idx.reverse()
    imgs=[imgs[i] for i in range(n) if remain_idx[i]]
    masks=[masks[i] for i in range(n) if remain_idx[i]]
    n=len(imgs)

    img_target=imgs[0].clone()
    for i in range(n-1,-1,-1):
        masks[i][masks[i]<masks_join]=0
        img_target[masks[i]>0]=imgs[i][masks[i]>0]

    # torchvision.utils.save_image(img_target, '/home/dengzijun/clip_transfer/github/output/target2.png', normalize=True, range=(-1, 1))
    # assert(0)
    img_target=img_target.unsqueeze(0)
    img_final,latent=project(orig_img,imgs,masks,img_target,g_ema,percept,projector_dir)
    return img_final,orig_img

def main(sentences):
    percept,g_ema,text_encoder,netG2,ps,clip_model=prepare_model()
    projector_dir=os.path.join(output_dir,"projector")
    attn_dir=os.path.join(output_dir,"features")

    parsed=ps.parse(sentences)

    txt = sentences[0]
    with torch.no_grad():
        tokenized_txt = clip.tokenize([txt]).to('cuda')
        txt_fts = clip_model.encode_text(tokenized_txt)
        txt_fts = txt_fts/txt_fts.norm(dim=-1, keepdim=True)
        c = torch.zeros((1, g_ema.c_dim), device='cuda')
        z = torch.randn((1, 512)).to('cuda')
        ws = g_ema.mapping(z, c)
        w_opt = ws[:,0,:]
        orig = g_ema.synthesis(ws, fts=txt_fts, noise_mode='const')
        

    # get feature images and attention images, match it to the origin image
    # parsed: {0: (-1, 'A bird'), 3: (1, 'a blue tail'), 7: (0, 'red feet'), ...}
    print(parsed)
    # assert(0)
    imgs,masks,orig_img=get_mask3(parsed,attn_dir,text_encoder,netG2,orig)

    img_target=imgs[0].clone()
    n=len(masks)

    masks_join=torch.zeros((3,256,256)).cuda()
    remain_idx=[]
    for i in range(n-1,-1,-1):
        masks[i]=transforms.ToTensor()(masks[i]).cuda()
        masks[i][masks[i]<0.1]=0
        m1,m2=masks[i]>0.1, masks[i]>masks_join
        m3,m4=masks[i]>0.2, masks_join>0.2
        ratio1=torch.sum(m1 & m2)/(torch.sum(m1).float()+1e-3)
        ratio2=torch.sum(m3 & m4)/(torch.sum(m3).float()+1e-3)
        print(ratio1,ratio2)
        remain=ratio1>=0.4 and ratio2<=0.9
        remain_idx.append(remain)
        if not remain:
            continue
        masks_join[masks[i]>masks_join]=masks[i][masks[i]>masks_join]

    remain_idx.reverse()
    imgs=[imgs[i] for i in range(n) if remain_idx[i]]
    masks=[masks[i] for i in range(n) if remain_idx[i]]
    n=len(imgs)

    img_target=imgs[0].clone()
    for i in range(n-1,-1,-1):
        masks[i][masks[i]<masks_join]=0
        img_target[masks[i]>0]=imgs[i][masks[i]>0]

    # torchvision.utils.save_image(img_target, '/home/dengzijun/clip_transfer/github/output/target2.png', normalize=True, range=(-1, 1))
    # assert(0)
    img_target=img_target.unsqueeze(0)
    img_final,latent=project2(orig_img,imgs,masks,img_target,g_ema,percept,projector_dir,w_opt,txt_fts,orig)
    return img_final,orig


def load_filenames(data_dir, split):
    filepath = '%s/%s/filenames.pickle' % (data_dir, split)
    with open(filepath, 'rb') as f:
        filenames = pickle.load(f)
    print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
    return filenames


def gen_origin_image(g_ema,sentence):
    code=get_code(sentence)
    code=torch.from_numpy(code).cuda()
    imgs, _ = g_ema([code.unsqueeze(0)], truncation=1)  
    return imgs



def generate_all(img_dir):
    text_filenames=load_filenames(cfg.DATA_DIR, 'test')
    percept,g_ema,text_encoder,netG2,ps,clip_model=prepare_model()

    for i in range(1000):
        print('{}/{}'.format(i+1,len(text_filenames)),end='\r')
        
        cap_path = '%s/text/%s.txt' % (cfg.DATA_DIR, text_filenames[i])
        caps=[]
        with open(cap_path, "r") as f:
            captions = f.read().encode('utf8').decode('utf8').split('\n')
            for cap in captions:
                if len(cap) == 0:
                    continue
                caps.append(cap)

        try:
            img,orig=_main(percept,g_ema,text_encoder,netG2,ps,clip_model,[caps[0]])
            # img=gen_origin_image(g_ema,sentence)
            # idx=i*10+j
            torchvision.utils.save_image(orig,os.path.join(img_dir,f"{str(i).zfill(6)}-ori.png"),
                    nrow=1,normalize=True,range=(-1, 1),)

            img.save(os.path.join(img_dir,f"{str(i).zfill(6)}.png"))
        except Exception:
            traceback.print_stack()
            print(caps)
            print(i)
            print('-'*30)


    print()


if __name__== '__main__':
    import random
    manualSeed = random.randint(1, 10000)
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    sentences = [
        'A bird with a blue belly and a green crown, the tail is yellow and the feet are red.',
        # 'A bird with a orange belly and a black crown, the tail is black.'
    ]

    # sentences = [
    #     'a white flower.'
    # ]

    # sentences=get_sentences(os.path.join(data_dir,text_paths[i]))
    if args['space']=='lafite':
        img,_=main(sentences)
    else:
        # stylegan2 latent space
        img,_=_main(sentences)
    
    # img,_=gen_example(sentences[0])
    img.save('output/0.png')
