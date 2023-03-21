import os
import torch
import torch.nn as nn
from torchvision import models,transforms
from torch.autograd import Variable
import numpy as np
from PIL import Image

 
class icloss(nn.Module):
    def __init__(self):
        super(icloss, self).__init__()
        self.net = models.inception_v3(pretrained=True)
        self.net.eval()
        self.net.cuda()
        self.pad=nn.ZeroPad2d(padding=(22,22,21,21))
 
    def extract_feats(self, input):
        x = self.pad(input)
        x = self.net.Conv2d_1a_3x3(x)
        x = self.net.Conv2d_2a_3x3(x)
        x = self.net.Conv2d_2b_3x3(x)
        x = self.net.maxpool1(x)
        x = self.net.Conv2d_3b_1x1(x)
        x = self.net.Conv2d_4a_3x3(x)
        x = self.net.maxpool2(x)
        x = self.net.Mixed_5b(x)
        x = self.net.Mixed_5c(x)
        x = self.net.Mixed_5d(x)
        x = self.net.Mixed_6a(x)
        x = self.net.Mixed_6b(x)
        x = self.net.Mixed_6c(x)
        x = self.net.Mixed_6d(x)
        x = self.net.Mixed_6e(x)
        x = self.net.Mixed_7a(x)
        x = self.net.Mixed_7b(x)
        x = self.net.Mixed_7c(x)
        x = self.net.avgpool(x)
        x = torch.flatten(x, 1)
        return x
    
    def forward(self, y_hat, y):
        n_samples = y.shape[0]
        y_feats = self.extract_feats(y)  # Otherwise use the feature from there
        y_hat_feats = self.extract_feats(y_hat)
        y_feats = y_feats.detach()
        loss = 0
        count = 0
        for i in range(n_samples):
            diff_target = torch.cosine_similarity(y_hat_feats,y_feats)
            loss += 1 - diff_target
            count += 1

        return loss / count
 
