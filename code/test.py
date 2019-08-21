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
from opts import TrainOptions
from matplotlib import pyplot as plt
from torch.optim import Adam
from tqdm import tqdm
import math
import numpy as np
import cv2
import os

"""
    Author: Wei Wang
"""

def psnr2(img1, img2):
   mse = np.mean( (img1/255. - img2/255.) ** 2 )
   if mse < 1.0e-10:
      return 100
   PIXEL_MAX = 1
   return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

scene_directory = 'F:\project\AHDRNet\data\Test\PAPER\ManStanding'
def inference():
    # Load the image
    # Read Expo times in scene
    expoTimes = ReadExpoTimes(os.path.join(scene_directory, 'exposure.txt'))
    # Read Image in scene
    imgs = ReadImages(list_all_files_sorted(scene_directory, '.tif'))
    # Read label
    label = ReadLabel(scene_directory)
    # inputs-process
    pre_img0 = LDR_to_HDR(imgs[0], expoTimes[0], 2.2)
    pre_img1 = LDR_to_HDR(imgs[1], expoTimes[1], 2.2)
    pre_img2 = LDR_to_HDR(imgs[2], expoTimes[2], 2.2)
    output0 = np.concatenate((imgs[0], pre_img0), 2)
    output1 = np.concatenate((imgs[1], pre_img1), 2)
    output2 = np.concatenate((imgs[2], pre_img2), 2)
    # label-process
    label = range_compressor(label)*255.0

    im1 = torch.Tensor(output0).cuda()
    im1 = torch.unsqueeze(im1, 0).permute(0, 3, 1, 2)

    im2 = torch.Tensor(output1).cuda()
    im2 = torch.unsqueeze(im2, 0).permute(0, 3, 1, 2)

    im3 = torch.Tensor(output2).cuda()
    im3 = torch.unsqueeze(im3, 0).permute(0, 3, 1, 2)

    # Load the pre-trained model
    model = AHDRNet().cuda()
    model.eval()
    model.load_state_dict(torch.load('./Model/2300.pkl'))

    # Run
    with torch.no_grad():
        # Forward
        pre = model(im1, im2, im3)
    pre = torch.clamp(pre, 0., 1.)
    pre = pre.data[0].cpu().numpy()
    pre = np.clip(pre * 255.0, 0., 255.)
    pre = pre.transpose(1, 2, 0)
    #pre1=cv2.cvtColor(pre, cv2.COLOR_RGB2BGR)
    p=psnr2(pre,label)
    print(p)
    #cv2.imwrite('./recover/PeopleTalking/out.png', pre)
    #cv2.imwrite('./recover/PeopleTalking/label.png', label)


if __name__ == '__main__':
    inference()
