import numpy as np
import os, glob
import cv2
import imageio
from math import log10
import torch
from skimage.measure.simple_metrics import compare_psnr
imageio.plugins.freeimage.download()


def ReadExpoTimes(fileName):
    return np.power(2, np.loadtxt(fileName))


def list_all_files_sorted(folderName, extension=""):
    return sorted(glob.glob(os.path.join(folderName, "*" + extension)))


def ReadImages(fileNames):
    imgs = []
    for imgStr in fileNames:
        img = cv2.imread(imgStr, -1)

        # equivalent to im2single from Matlab
        img = img / 2 ** 16
        img = np.float32(img)

        img.clip(0, 1)

        imgs.append(img)
    return np.array(imgs)


def ReadLabel(fileName):
    label = imageio.imread(os.path.join(fileName, 'HDRImg.hdr'), 'hdr')
    label = label[:, :, [2, 1, 0]]  ##cv2
    return label


def LDR_to_HDR(imgs, expo, gamma):
    return (imgs ** gamma) / expo


def range_compressor(x):
    return (np.log(1 + 5000 * x)) / np.log(1 + 5000)

def psnr(x, target):
    sqrdErr = np.mean((x - target) ** 2)
    return 10 * log10(1/sqrdErr)

def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])
