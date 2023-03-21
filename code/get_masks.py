import argparse
import math
import os,sys
from click import style

from numpy.lib import utils
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
from PIL import Image

import torch
import torchvision
from torch import optim
from tqdm import tqdm

from stylegan2.model import Generator, Discriminator
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from model import *
from miscc.config import cfg,cfg_from_file
from nltk.tokenize import RegexpTokenizer
from sentence_parser import parser
from miscc.utils import build_super_images, build_super_images2, load_dataset_dict_info, get_code
import rembg.bg as bg
from miscc.losses import cosine_similarity
from stylegan2 import dnnlib
from stylegan2 import legacy
from skimage.measure import compare_ssim as ssim

if cfg.CONFIG_NAME=='':
    cfg_from_file('code/cfg/eval_bird.yml')
ixtoword, wordtoix, n_words=load_dataset_dict_info()
discriminator=Discriminator(256,1)
stylegan_ckpt=cfg.STYLEGAN
discriminator.load_state_dict(torch.load(stylegan_ckpt)["d"])
discriminator=discriminator.cuda()

# import clip
# model, preprocess = clip.load("ViT-B/32", device='cuda')

def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()
    return loss

class G_NET2(nn.Module):
    def __init__(self):
        super(G_NET2, self).__init__()
        ngf = cfg.GAN.GF_DIM
        nef = cfg.TEXT.EMBEDDING_DIM
        ncf = cfg.GAN.CONDITION_DIM
        self.ca_net = CA_NET()

        if cfg.TREE.BRANCH_NUM > 0:
            self.h_net1 = INIT_STAGE_G(ngf * 16, ncf)
            self.img_net1 = GET_IMAGE_G(ngf)
        # gf x 64 x 64
        if cfg.TREE.BRANCH_NUM > 1:
            self.h_net2 = NEXT_STAGE_G(ngf, nef, ncf, 64)
            self.img_net2 = GET_IMAGE_G(ngf)
        if cfg.TREE.BRANCH_NUM > 2:
            self.h_net3 = NEXT_STAGE_G(ngf, nef, ncf, 128)
            self.img_net3 = GET_IMAGE_G(ngf)

    def forward(self, h_code1, sent_emb, word_embs, mask, cap_lens):
        """
            :param z_code: batch x cfg.GAN.Z_DIM
            :param sent_emb: batch x cfg.TEXT.EMBEDDING_DIM
            :param word_embs: batch x cdf x seq_len
            :param mask: batch x seq_len
            :return:
        """
        fake_imgs = []
        att_maps = []
        c_code, mu, logvar = self.ca_net(sent_emb)

        if cfg.TREE.BRANCH_NUM > 1:
            h_code2, att1 = self.h_net2(h_code1, c_code, word_embs, mask, cap_lens)
            fake_img2 = self.img_net2(h_code2)
            fake_imgs.append(fake_img2)
            if att1 is not None:
                att_maps.append(att1)
        if cfg.TREE.BRANCH_NUM > 2:
            h_code3, att2 = self.h_net3(h_code2, c_code, word_embs, mask, cap_lens)
            fake_img3 = self.img_net3(h_code3)
            fake_imgs.append(fake_img3)
            if att2 is not None:
                att_maps.append(att2)
        return fake_imgs, att_maps, mu, logvar


def prepare():
    ixtoword, wordtoix,n_words=load_dataset_dict_info()
    # Build and load the generator
    text_encoder = RNN_ENCODER(n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
    state_dict = torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
    text_encoder.load_state_dict(state_dict)
    text_encoder = text_encoder.cuda()
    text_encoder.eval()

    # the path to save generated images
    netG2 = G_NET2()
    netG = G_NET()

    model_dir = cfg.TRAIN.NET_G
    state_dict = \
        torch.load(model_dir, map_location=lambda storage, loc: storage)
    netG.load_state_dict(state_dict)
    netG.cuda()
    netG.eval()

    netG2.load_state_dict(state_dict)
    netG2.cuda()
    netG2.eval()

    return ixtoword,text_encoder,netG,netG2

def preprocess(sent):
    captions = []
    cap_lens = []

    sent = sent.replace("\ufffd\ufffd", " ")
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(sent.lower())
    if len(tokens) == 0:
        print('sent', sent)

    rev = []
    for t in tokens:
        t = t.encode('ascii', 'ignore').decode('ascii')
        if len(t) > 0 and t in wordtoix:
            rev.append(wordtoix[t])
    captions.append(rev)
    cap_lens.append(len(rev))

    max_len = np.max(cap_lens)
    sorted_indices = np.argsort(cap_lens)[::-1]
    cap_lens = np.asarray(cap_lens)
    cap_lens = cap_lens[sorted_indices]
    cap_array = np.zeros((len(captions), max_len), dtype='int64')
    for i in range(len(captions)):
        idx = sorted_indices[i]
        cap = captions[idx]
        c_len = len(cap)
        cap_array[i, :c_len] = cap
    data_dic = [cap_array, cap_lens, sorted_indices]
    return data_dic

def get_data_dic(sentences):
    captions = []
    cap_lens = []
    pos = []
    noun_words = []

    for p,n,sent in sentences:
        sent='This bird '+sent
        sent = sent.replace("\ufffd\ufffd", " ")
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(sent.lower())
        if len(tokens) == 0:
            print('sent', sent)

        rev = []
        for t in tokens:
            t = t.encode('ascii', 'ignore').decode('ascii')
            if len(t) > 0 and t in wordtoix:
                rev.append(wordtoix[t])
        captions.append(rev)
        cap_lens.append(len(rev))

        pos.append(p+2)
        for i in range(len(n)):
            n[i]+=2
        noun_words.append(n)

    max_len = np.max(cap_lens)
    sorted_indices = np.argsort(cap_lens)[::-1]
    cap_lens = np.asarray(cap_lens)
    cap_lens = cap_lens[sorted_indices]
    cap_array = np.zeros((len(captions), max_len), dtype='int64')
    new_pos=[]
    new_noun_words=[]
    for i in range(len(captions)):
        idx = sorted_indices[i]
        cap = captions[idx]
        c_len = len(cap)
        cap_array[i, :c_len] = cap
        new_pos.append(pos[idx])
        new_noun_words.append(noun_words[idx])

    data_dic = [cap_array, cap_lens, sorted_indices, new_pos, new_noun_words]
    return data_dic

def get_noise(seed):
    noise=seed.mean()*torch.randn_like(seed)+seed
    noise=noise.cuda()
    return noise

def get_mask(sentence):
    save_dir='output/features'
    ixtoword,text_encoder,netG,netG2=prepare()
    data_dic1=preprocess('a little white bird')
    captions, cap_lens, sorted_indices = data_dic1
    with torch.no_grad():
        batch_size = captions.shape[0]
        nz = cfg.GAN.Z_DIM
        captions = Variable(torch.from_numpy(captions))
        cap_lens = Variable(torch.from_numpy(cap_lens))

        captions = captions.cuda()
        cap_lens = cap_lens.cuda()

        noise = Variable(torch.FloatTensor(batch_size, nz))
        noise = noise.cuda()
        cap_lens_np = cap_lens.cpu().data.numpy()
        #######################################################
        # (1) Extract text embeddings
        ######################################################
        hidden = text_encoder.init_hidden(batch_size)
        # words_embs: batch_size x nef x seq_len
        # sent_emb: batch_size x nef
        words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
        mask = (captions == 0)
        #######################################################
        # (2) Generate fake images
        ######################################################
        noise.data.normal_(0, 1)

        c_code, mu, logvar = netG.ca_net(sent_emb)
        h_code1 = netG.h_net1(noise, c_code)
        fake_img1 = netG.img_net1(h_code1)

        torchvision.utils.save_image(fake_img1, os.path.join(save_dir,"64x64.png"), normalize=True, range=(-1, 1))
    
    h_code1=h_code1.repeat(4,1,1,1)
    ps=parser()
    parsed=ps.parse(sentence)
    caps=list(parsed.values())
    data_dic=get_data_dic(caps)
    captions, cap_lens, sorted_indices,new_pos = data_dic
    imgs=[]
    with torch.no_grad():
        batch_size = captions.shape[0]
        nz = cfg.GAN.Z_DIM
        captions = Variable(torch.from_numpy(captions))
        cap_lens = Variable(torch.from_numpy(cap_lens))

        captions = captions.cuda()
        cap_lens = cap_lens.cuda()

        noise = Variable(torch.FloatTensor(batch_size, nz))
        noise = noise.cuda()
        #######################################################
        # (1) Extract text embeddings
        ######################################################
        hidden = text_encoder.init_hidden(batch_size)
        # words_embs: batch_size x nef x seq_len
        # sent_emb: batch_size x nef
        words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
        mask = (captions == 0)
        #######################################################
        # (2) Generate fake images
        ######################################################
        noise.data.normal_(0, 1)
        fake_imgs, attention_maps, _, _ = netG2(h_code1, sent_emb, words_embs, mask, cap_lens)
        # G attention
        cap_lens_np = cap_lens.cpu().data.numpy()

        for k in range(len(new_pos)):
            im = fake_imgs[1][k].data.cpu().numpy()
            im = (im + 1.0) * 127.5
            im = im.astype(np.uint8)
            # print('im', im.shape)
            im = np.transpose(im, (1, 2, 0))
            # print('im', im.shape)
            im = Image.fromarray(im)
            fullpath = os.path.join(save_dir,'output{}.png'.format(k))
            im.save(fullpath)
            imgs.append(im)

        attn_maps = attention_maps[0]
        att_sze = attn_maps.size(2)
        img_set, masks, sentences = \
            build_super_images2(fake_imgs[1].cpu(),
                                captions,
                                cap_lens_np, ixtoword,
                                attn_maps, att_sze,
                                topK=5,save_mask=new_pos,save_dir=save_dir)
        # print(sentences)
        if img_set is not None:
            im = Image.fromarray(img_set)
            fullpath = os.path.join(save_dir,'attn.png')
            im.save(fullpath)
    return imgs,masks

def _get_mask(sentence):
    attn_dir = 'output/features'
    ixtoword,text_encoder,netG,netG2=prepare()
    ps=parser()
    parsed=ps.parse([sentence])

    # get feature images and attention images, match it to the origin image
    # parsed: {0: (-1, 'A bird'), 3: (1, 'a blue tail'), 7: (0, 'red feet'), ...}
    print(parsed)
    # assert(0)
    imgs,masks,orig_img=get_mask3(parsed,attn_dir,text_encoder,netG2)

def get_h_code2(sentence,text_encoder,netG2):
    sentences=[sentence]*12
    data_dic1=preprocess2(sentences)
    captions, cap_lens, sorted_indices = data_dic1
    with torch.no_grad():
        batch_size = captions.shape[0]
        nz = cfg.GAN.Z_DIM
        captions = Variable(torch.from_numpy(captions))
        cap_lens = Variable(torch.from_numpy(cap_lens))

        captions = captions.cuda()
        cap_lens = cap_lens.cuda()

        noise = Variable(torch.FloatTensor(batch_size, nz))
        noise = noise.cuda()
        cap_lens_np = cap_lens.cpu().data.numpy()
        #######################################################
        # (1) Extract text embeddings
        ######################################################
        hidden = text_encoder.init_hidden(batch_size)
        # words_embs: batch_size x nef x seq_len
        # sent_emb: batch_size x nef
        words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
        mask = (captions == 0)
        #######################################################
        # (2) Generate fake images
        ######################################################
        noise.data.normal_(0, 1)

        c_code, mu, logvar = netG2.ca_net(sent_emb)
        h_code1 = netG2.h_net1(noise, c_code)
        # fake_imgs = netG2.img_net1(h_code1)
        # fake_imgs=nn.Upsample((256,256))(fake_imgs)
        h_code2, _ = netG2.h_net2(h_code1, c_code, words_embs, mask, cap_lens)
        h_code3, _ = netG2.h_net3(h_code2, c_code, words_embs, mask, cap_lens)
        fake_imgs = netG2.img_net3(h_code3)
    
    torchvision.utils.save_image(fake_imgs, 'output/features/select.png', nrow=4, normalize=True, range=(-1, 1))
    masks=bg.get_mask(fake_imgs)
    # masks=masks.repeat(1,3,1,1)
    # torchvision.utils.save_image(masks, 'output/features/shape.png', nrow=4, normalize=True, range=(0, 1))
    # fake_imgs=bilateralFilter(fake_imgs,13)
    # torchvision.utils.save_image(fake_imgs, 'output/features/select.png', nrow=4, normalize=True, range=(-1, 1))

    # imgs2=bg.pytorch_remove(fake_imgs)
    # torchvision.utils.save_image(imgs2, 'output/features/shape.png', nrow=4, normalize=True, range=(-1, 1))
    fake_pred = discriminator(fake_imgs)
    # image = preprocess(fake_imgs).cuda()
    # image = nn.Upsample((224,224))(fake_imgs)
    # text = clip.tokenize(['a medium size bird']).cuda()

    code=get_code(sentence)
    latent=torch.from_numpy(code).unsqueeze(0).cuda()

    # with torch.no_grad():       
    #     logits_per_image, logits_per_text = model(image, text)
    #     probs = logits_per_text.softmax(dim=-1)

    # print(fake_pred)
    # best_i=torch.argmax(probs)
    best_i=torch.argmax(fake_pred)
    orig_img=fake_imgs[best_i]
    print(best_i)
    area=torch.sum(masks[best_i]>0.8)*(1/256/256)
    
    return h_code1[best_i].unsqueeze(0),orig_img,area

def get_mask2(caps,save_dir,text_encoder,netG2):
    data_dic=get_data_dic(caps)
    captions, cap_lens, sorted_indices, new_pos, new_noun_words = data_dic

    new_indices=np.zeros_like(sorted_indices)
    for i in range(len(new_indices)):
        new_indices[sorted_indices[i]]=i
    
    captions, cap_lens, new_pos=filter_info(captions, cap_lens, sorted_indices, new_pos, new_noun_words, text_encoder)

    max_len = np.max(cap_lens)
    sorted_indices = np.argsort(cap_lens)[::-1]
    cap_lens = cap_lens[sorted_indices]
    captions2=np.zeros((len(cap_lens),max_len), dtype='int64')
    pos2=[]
    for i in range(len(cap_lens)):
        idx=sorted_indices[i]
        captions2[i]=captions[idx][:max_len]
        pos2.append(new_pos[idx])

    for i in range(len(captions2)):
        cap=captions2[i]
        sentence = []
        for j in range(len(cap)):
            if cap[j] == 0:
                break
            word = ixtoword[cap[j]].encode('ascii', 'ignore').decode('ascii')
            sentence.append(word)
        print(sentence,pos2[i],cap_lens[i])
    # assert(0)

    area=0
    while area<0.1 or area>0.3:
        h_code1,orig_img,area=get_h_code2('this bird '+caps[0][2],text_encoder,netG2)
    # assert(0)

    h_code1=h_code1.repeat(len(pos2),1,1,1)
    imgs=[]
    with torch.no_grad():
        batch_size = captions2.shape[0]
        nz = cfg.GAN.Z_DIM
        captions = Variable(torch.from_numpy(captions2))
        cap_lens = Variable(torch.from_numpy(cap_lens))

        captions = captions.cuda()
        cap_lens = cap_lens.cuda()

        noise = Variable(torch.FloatTensor(batch_size, nz))
        noise = noise.cuda()
        #######################################################
        # (1) Extract text embeddings
        ######################################################
        hidden = text_encoder.init_hidden(batch_size)
        # words_embs: batch_size x nef x seq_len
        # sent_emb: batch_size x nef
        words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)

        mask = (captions == 0)
        #######################################################
        # (2) Generate fake images
        ######################################################
        noise.data.normal_(0, 1)
        fake_imgs, attention_maps, _, _ = netG2(h_code1, sent_emb, words_embs, mask, cap_lens)
        # G attention
        cap_lens_np = cap_lens.cpu().data.numpy()

        for k in range(len(pos2)):
            im = fake_imgs[-1][k].data.cpu().numpy()
            im = (im + 1.0) * 127.5
            im = im.astype(np.uint8)
            # print('im', im.shape)
            im = np.transpose(im, (1, 2, 0))
            # print('im', im.shape)
            im = Image.fromarray(im)
            fullpath = os.path.join(save_dir,'output{}.png'.format(k))
            im.save(fullpath)
            imgs.append(fake_imgs[-1][k])

        attn_maps = attention_maps[-1]
        att_sze = attn_maps.size(2)
        img_set, masks, sentences = \
            build_super_images2(fake_imgs[-1].cpu(),
                                captions,
                                cap_lens_np, ixtoword,
                                attn_maps, att_sze,
                                topK=5,save_mask=pos2,save_dir=save_dir)
        # print(sentences)
        if img_set is not None:
            im = Image.fromarray(img_set)
            fullpath = os.path.join(save_dir,'attn.png')
            im.save(fullpath)
    return imgs,masks,orig_img

def preprocess2(sentences):
    captions = []
    cap_lens = []

    for sent in sentences:
        sent = sent.replace("\ufffd\ufffd", " ")
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(sent.lower())
        if len(tokens) == 0:
            print('sent', sent)

        rev = []
        for t in tokens:
            t = t.encode('ascii', 'ignore').decode('ascii')
            if len(t) > 0 and t in wordtoix:
                rev.append(wordtoix[t])
        captions.append(rev)
        cap_lens.append(len(rev))

    max_len = np.max(cap_lens)
    sorted_indices = np.argsort(cap_lens)[::-1]
    cap_lens = np.asarray(cap_lens)
    cap_lens = cap_lens[sorted_indices]
    cap_array = np.zeros((len(captions), max_len), dtype='int64')
    for i in range(len(captions)):
        idx = sorted_indices[i]
        cap = captions[idx]
        c_len = len(cap)
        cap_array[i, :c_len] = cap
    data_dic = [cap_array, cap_lens, sorted_indices]
    return data_dic

def get_h_code(sentence,text_encoder,netG2,orig):
    sentences=[sentence]*20
    data_dic1=preprocess2(sentences)
    captions, cap_lens, sorted_indices = data_dic1
    with torch.no_grad():
        batch_size = captions.shape[0]
        nz = cfg.GAN.Z_DIM
        captions = Variable(torch.from_numpy(captions))
        cap_lens = Variable(torch.from_numpy(cap_lens))

        captions = captions.cuda()
        cap_lens = cap_lens.cuda()

        noise = Variable(torch.FloatTensor(batch_size, nz))
        noise = noise.cuda()
        cap_lens_np = cap_lens.cpu().data.numpy()
        #######################################################
        # (1) Extract text embeddings
        ######################################################
        hidden = text_encoder.init_hidden(batch_size)
        # words_embs: batch_size x nef x seq_len
        # sent_emb: batch_size x nef
        words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
        mask = (captions == 0)
        #######################################################
        # (2) Generate fake images
        ######################################################
        noise.data.normal_(0, 1)

        c_code, mu, logvar = netG2.ca_net(sent_emb)
        h_code1 = netG2.h_net1(noise, c_code)
        # fake_imgs = netG2.img_net1(h_code1)
        # fake_imgs=nn.Upsample((256,256))(fake_imgs)
        h_code2, _ = netG2.h_net2(h_code1, c_code, words_embs, mask, cap_lens)
        h_code3, _ = netG2.h_net3(h_code2, c_code, words_embs, mask, cap_lens)
        fake_imgs = netG2.img_net3(h_code3)
    
    
    torchvision.utils.save_image(fake_imgs, 'output/features/select.png', nrow=4, normalize=True, range=(-1, 1))
    torchvision.utils.save_image(orig, 'output/features/orig.png', nrow=4, normalize=True, range=(-1, 1))
    masks=bg.get_mask(fake_imgs)
    morig=bg.get_mask(orig)
    morig=morig.squeeze(0).squeeze(0).detach().cpu().numpy()
    max_value=0
    max_idx=-1
    for idx,fake_img in enumerate(masks):
        fake2=fake_img.squeeze(0).detach().cpu().numpy()
        ss=ssim(fake2,morig)
        if ss>max_value:
            max_value=ss
            max_idx=idx
    # assert(0)

    # code=get_code(sentence)
    # latent=torch.from_numpy(code).unsqueeze(0).cuda()
    # if seed is not None:
    #     noise=get_noise(seed)

    # # with torch.no_grad():       
    # #     logits_per_image, logits_per_text = model(image, text)
    # #     probs = logits_per_text.softmax(dim=-1)

    # # print(fake_pred)
    # # best_i=torch.argmax(probs)
    # best_i=torch.argmax(fake_pred)
    # orig_img=fake_imgs[best_i]
    # print(best_i)
    # area=torch.sum(masks[best_i]>0.8)*(1/256/256)
    # if seed is not None:
    #     orig_img=g_ema(latent,noise)
    
    # return h_code1[best_i].unsqueeze(0),orig_img,area
    return h_code1[max_idx].unsqueeze(0),fake_imgs[max_idx],torch.sum(masks[0]>0.8)*(1/256/256)

def filter_info(captions, cap_lens, sorted_indices, pos, noun_words, text_encoder):
    embs=[]
    with torch.no_grad():
        batch_size = captions.shape[0]
        nz = cfg.GAN.Z_DIM
        captions = Variable(torch.from_numpy(captions))
        cap_lens = Variable(torch.from_numpy(cap_lens))

        captions = captions.cuda()
        cap_lens = cap_lens.cuda()

        noise = Variable(torch.FloatTensor(batch_size, nz))
        noise = noise.cuda()
        #######################################################
        # (1) Extract text embeddings
        ######################################################
        hidden = text_encoder.init_hidden(batch_size)
        # words_embs: batch_size x nef x seq_len
        # sent_emb: batch_size x nef
        words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)

    max_cap_len=captions.size(1)
    remain_idx=np.zeros(batch_size,dtype=bool)
    for i in range(batch_size):
        b=0
        idx = sorted_indices[i]
        noun_words_i=noun_words[idx].copy()
        for n_pos in noun_words_i:
            n_pos=n_pos+b
            emb=words_embs[idx,:,n_pos]
            word_id=captions[idx][n_pos].item()
            
            ok=1
            for x,x_id in embs:
                if x_id == word_id or cosine_similarity(x,emb,dim=0)>0.75:
                    ok=0
                    break
            if ok:
                embs.append((emb,word_id))
            else:
                noun_words[idx].remove(n_pos-b)
                new_cap=torch.zeros_like(captions[idx])
                new_cap[:-1] = captions[idx][torch.arange(max_cap_len)!=n_pos]
                captions[idx]=new_cap
                cap_lens[idx]-=1
                b-=1
                if n_pos-b<pos[idx]:
                    pos[idx]-=1
        remain_idx[idx]=(len(noun_words[idx])!=0)

    captions = captions[remain_idx].data.cpu().numpy()
    cap_lens = cap_lens[remain_idx].data.cpu().numpy()
    pos=[pos[i] for i in range(len(pos)) if remain_idx[i]]
    assert len(pos)==captions.shape[0]
    return captions, cap_lens, pos

def get_mask3(caps,save_dir,text_encoder,netG2,orig):
    data_dic=get_data_dic(caps)
    captions, cap_lens, sorted_indices, new_pos, new_noun_words = data_dic

    new_indices=np.zeros_like(sorted_indices)
    for i in range(len(new_indices)):
        new_indices[sorted_indices[i]]=i
    
    captions, cap_lens, new_pos=filter_info(captions, cap_lens, sorted_indices, new_pos, new_noun_words, text_encoder)

    max_len = np.max(cap_lens)
    sorted_indices = np.argsort(cap_lens)[::-1]
    cap_lens = cap_lens[sorted_indices]
    captions2=np.zeros((len(cap_lens),max_len), dtype='int64')
    pos2=[]
    for i in range(len(cap_lens)):
        idx=sorted_indices[i]
        captions2[i]=captions[idx][:max_len]
        pos2.append(new_pos[idx])

    for i in range(len(captions2)):
        cap=captions2[i]
        sentence = []
        for j in range(len(cap)):
            if cap[j] == 0:
                break
            word = ixtoword[cap[j]].encode('ascii', 'ignore').decode('ascii')
            sentence.append(word)
        print(sentence,pos2[i],cap_lens[i])
    # assert(0)

    area=0
    while area<0.1 or area>0.3:
        h_code1,orig_img,area=get_h_code('this bird '+caps[0][2],text_encoder,netG2,orig)
    # assert(0)

    h_code1=h_code1.repeat(len(pos2),1,1,1)
    imgs=[]
    with torch.no_grad():
        batch_size = captions2.shape[0]
        nz = cfg.GAN.Z_DIM
        captions = Variable(torch.from_numpy(captions2))
        cap_lens = Variable(torch.from_numpy(cap_lens))

        captions = captions.cuda()
        cap_lens = cap_lens.cuda()

        noise = Variable(torch.FloatTensor(batch_size, nz))
        noise = noise.cuda()
        #######################################################
        # (1) Extract text embeddings
        ######################################################
        hidden = text_encoder.init_hidden(batch_size)
        # words_embs: batch_size x nef x seq_len
        # sent_emb: batch_size x nef
        words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)

        mask = (captions == 0)
        #######################################################
        # (2) Generate fake images
        ######################################################
        noise.data.normal_(0, 1)
        fake_imgs, attention_maps, _, _ = netG2(h_code1, sent_emb, words_embs, mask, cap_lens)
        # G attention
        cap_lens_np = cap_lens.cpu().data.numpy()

        for k in range(len(pos2)):
            im = fake_imgs[-1][k].data.cpu().numpy()
            im = (im + 1.0) * 127.5
            im = im.astype(np.uint8)
            # print('im', im.shape)
            im = np.transpose(im, (1, 2, 0))
            # print('im', im.shape)
            im = Image.fromarray(im)
            fullpath = os.path.join(save_dir,'output{}.png'.format(k))
            im.save(fullpath)
            imgs.append(fake_imgs[-1][k])

        attn_maps = attention_maps[-1]
        att_sze = attn_maps.size(2)
        img_set, masks, sentences = \
            build_super_images2(fake_imgs[-1].cpu(),
                                captions,
                                cap_lens_np, ixtoword,
                                attn_maps, att_sze,
                                topK=5,save_mask=pos2,save_dir=save_dir)
        # print(sentences)
        if img_set is not None:
            im = Image.fromarray(img_set)
            fullpath = os.path.join(save_dir,'attn.png')
            im.save(fullpath)
    return imgs,masks,orig_img


if __name__ == "__main__":
    import random
    manualSeed = random.randint(1, 10000)
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    _get_mask('A bird with a yellow tail, red feet, white wings and a gray crown')
    print()

