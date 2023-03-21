import torch
from model import RNN_ENCODER,CNN_ENCODER
from miscc.losses import words_loss,sent_loss
from nltk.tokenize import RegexpTokenizer
import numpy as np
from torch.autograd import Variable
from miscc.config import cfg, cfg_from_file
from miscc.utils import load_dataset_dict_info
cfg_from_file('/home/dengzijun/clip_transfer/github/code/cfg/bird_text_mapper.yml')

class DAMSMLoss(torch.nn.Module):

    def __init__(self,batch_size=1):
        super(DAMSMLoss, self).__init__()
        self.wordtoix, n_words=load_dataset_dict_info()
        self.batch_size=batch_size
        self.text_encoder = RNN_ENCODER(n_words, nhidden=256).cuda()
        self.image_encoder = CNN_ENCODER(256).cuda()

        path='/home/dengzijun/clip_transfer/github/DAMSMencoders/bird/text_encoder200.pth'
        state_dict = torch.load(path)
        self.text_encoder.load_state_dict(state_dict)

        name = path.replace('text_encoder', 'image_encoder')
        state_dict = torch.load(name)
        self.image_encoder.load_state_dict(state_dict)

        self.labels = Variable(torch.LongTensor(range(batch_size)))
        self.labels = self.labels.cuda()

    def preprocess(self,sent):
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
            if len(t) > 0 and t in self.wordtoix:
                rev.append(self.wordtoix[t])
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

        captions = Variable(torch.from_numpy(cap_array))
        cap_lens = Variable(torch.from_numpy(cap_lens))

        captions = captions.cuda()
        cap_lens = cap_lens.cuda()
        return captions, cap_lens

    def forward(self, image, text):
        loss=0

        caption,cap_len=self.preprocess(text)
        hidden = self.text_encoder.init_hidden(self.batch_size)

        words_emb, sent_emb = self.text_encoder(caption, cap_len, hidden)
        words_features, sent_code = self.image_encoder(image)

        w_loss0, w_loss1, attn = words_loss(words_features, words_emb, self.labels,
                                            cap_len, None, self.batch_size)
        loss += (w_loss0 + w_loss1).data

        s_loss0, s_loss1 = \
            sent_loss(sent_code, sent_emb, self.labels, None, self.batch_size)
        loss += (s_loss0 + s_loss1).data
        return loss