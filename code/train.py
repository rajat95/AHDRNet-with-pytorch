import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import torch.utils.data as Data
import torch.autograd as autograd
from utils import *
from model import *
from datsetprocess import *
from math import log10


# init
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


# params
BATCH_SIZE = 8
LR = 1e-3

train_data = MyDataset(scene_directory='./data/Training')
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)

generator = AHDRNet().cuda()
L1_loss = nn.L1Loss()

# Initialize weights
# generator.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=LR)

Tensor = torch.cuda.FloatTensor
# ----------
#  Training
# ----------
for epoch in range(0, 10000):

    for step, sample in enumerate(train_loader):
        batch_x1, batch_x2, batch_x3, batch_x4 = sample['input1'], sample['input2'], sample['input3'], sample['label']
        batch_x1, batch_x2, batch_x3, batch_x4 = Variable(batch_x1).cuda(), Variable(batch_x2).cuda(), Variable(batch_x3).cuda(), Variable(batch_x4).cuda()

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Generate a batch of images
        pre = generator(batch_x1, batch_x2, batch_x3)

        # calculate PSNR
        PSNR = batch_PSNR(torch.clamp(pre, 0., 1.), batch_x4, 1.)
        g_loss = L1_loss(pre, batch_x4)

        g_loss.backward()
        optimizer_G.step()

        print("[Epoch %d][G loss: %7f][PSNR : %7f]" % (epoch, g_loss.item(), PSNR))
    if (epoch % 100 == 0):
        torch.save(generator.state_dict(), './Model/%d.pkl' % (epoch))
