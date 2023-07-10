import numpy as np
import cv2
from skimage import transform as trans

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import math

from scipy import stats as st

# 标准脸的关键点
arcface_src = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041] ], dtype=np.float32 )

def estimate_norm(lmk, image_size):
    assert lmk.shape==(5,2)
    tform = trans.SimilarityTransform()
    _src = float(image_size)/112 * arcface_src
    tform.estimate(lmk, _src)
    M = tform.params[0:2,:] # 估计变换矩阵M
    return M

def norm_crop(img, landmark, image_size=112, mode='arcface'):
    M = estimate_norm(landmark, image_size)
    warped = cv2.warpAffine(img,M, (image_size, image_size), borderValue = 0.0) # 将倾斜的人脸与标准脸对齐
    return warped, M

log_dir = Path('./')


class logSaver():
    def __init__(self, logname):
        self.logname = logname
        sys.stdout.flush()
        sys.stderr.flush()
        if self.logname == None:
            self.logpath_out = os.devnull
            self.logpath_err = os.devnull
        else:
            self.logpath_out = logname + "_out.log"
            self.logpath_err = logname + "_err.log"
        self.logfile_out = os.open(self.logpath_out, os.O_WRONLY | os.O_TRUNC | os.O_CREAT)
        self.logfile_err = os.open(self.logpath_err, os.O_WRONLY | os.O_TRUNC | os.O_CREAT)

    def __enter__(self):
        self.orig_stdout = os.dup(1)
        self.orig_stderr = os.dup(2)
        self.new_stdout = os.dup(1)
        self.new_stderr = os.dup(2)
        os.dup2(self.logfile_out, 1)
        os.dup2(self.logfile_err, 2)
        sys.stdout = os.fdopen(self.new_stdout, 'w')
        sys.stderr = os.fdopen(self.new_stderr, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.flush()
        sys.stderr.flush()

        os.dup2(self.orig_stdout, 1)
        os.dup2(self.orig_stderr, 2)
        os.close(self.orig_stdout)
        os.close(self.orig_stderr)

        os.close(self.logfile_out)
        os.close(self.logfile_err)

def get_max_grad_mask(grad, kpss, mask_size, k=3):
    t_grad = grad.detach().clone()
    t_grad = t_grad.squeeze().permute(1, 2, 0).cpu().numpy()
    g_max, g_min = t_grad.max(), t_grad.min()
    eps = 1e-7
    t_grad = (t_grad - g_min) / (g_max - g_min + eps)
    cam = np.uint8(t_grad * 255)

    mask = []
    avg_t_grad = []
    gray = cv2.cvtColor(cam, cv2.COLOR_RGB2GRAY)
    for i in range(5):
        sub_mask = np.zeros_like(gray)
        x, y = int(kpss[0][i][0]), int(kpss[0][i][1])
        x1 = max(0, x - mask_size)
        x2 = min(x + mask_size, 112)
        y1 = max(0, y - mask_size)
        y2 = min(y + mask_size, 112)
        sub_mask[x1:x2, y1:y2] = 1
        mask.append(sub_mask)
        area = (gray * sub_mask).sum()
        sub_avg_t_grad = area / sub_mask.sum()
        avg_t_grad.append(sub_avg_t_grad)

    avg_t_grad = torch.from_numpy(np.array(avg_t_grad))
    indices = torch.topk(avg_t_grad, k=k)[1].numpy()
    final_mask = np.zeros_like(mask[0])
    for index in indices:
        final_mask = final_mask + mask[index]
    return torch.from_numpy(final_mask)

class alphaScheduler:
    def __init__(self, min_alpha, max_alpha, T_max):
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.T_max = T_max

    def step(self, epoch):
        alpha = self.min_alpha + 0.5 * (self.max_alpha - self.min_alpha) * (1 + math.cos(epoch / self.T_max * math.pi))
        return alpha

class kernelGenerator:
    def __init__(self, kernel_name):
        self.kernel_name = kernel_name
        self.len_kernel = 3
        self.nsig = 3

    def kernel_generation(self):
        if self.kernel_name == 'gaussian':
            kernel = self.gkern(self.len_kernel, self.nsig).astype(np.float32)
        elif self.kernel_name == 'linear':
            kernel = self.lkern(self.len_kernel).astype(np.float32)
        elif self.kernel_name == 'uniform':
            kernel = self.ukern(self.len_kernel).astype(np.float32)
        else:
            raise NotImplementedError

        stack_kernel = np.stack([kernel, kernel, kernel])
        stack_kernel = np.expand_dims(stack_kernel, 1)
        return stack_kernel

    def gkern(self, kernlen=15, nsig=3):
        """Returns a 2D Gaussian kernel array."""
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def ukern(self, kernlen=15):
        kernel = np.ones((kernlen, kernlen)) * 1.0 / (kernlen * kernlen)
        return kernel

    def lkern(self, kernlen=15):
        kern1d = 1 - np.abs(np.linspace((-kernlen + 1) / 2, (kernlen - 1) / 2, kernlen) / (kernlen + 1) * 2)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel