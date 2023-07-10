import os
import os.path as osp
from scrfd import SCRFD
import numpy as np
import random
import torch
import glob
import torch.nn.functional as F
import cv2
import torch.nn as nn
from tqdm import tqdm
from utils import norm_crop, alphaScheduler
import iresnet

class PyFAT:
    def __init__(self, N=10):
        os.environ['PYTHONHASHSEED'] = str(3407)
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        torch.manual_seed(3407)
        np.random.seed(3407)
        random.seed(3407)
        self.alphaScheduler = alphaScheduler(min_alpha=0.5/255, max_alpha=2/255, T_max=10)
        self.num_iters = 30
        self.device = torch.device('cuda')
        self.mask_size = 8
        self.is_cuda = False


    def load(self, assets_path):
        detector = SCRFD(model_file=osp.join(assets_path, 'det_10g.onnx'))
        ctx_id = -1 if not self.is_cuda else 0
        detector.prepare(ctx_id, det_thresh=0.5, input_size=(160, 160))
        self.detector = detector

        model = iresnet.iresnet100()
        weight = osp.join(assets_path, 'glint360k_r100.pth')
        model.load_state_dict(torch.load(weight, map_location=self.device))
        model.eval().to(self.device)
        self.model1 = model

        model = iresnet.iresnet50()
        weight = osp.join(assets_path, 'w600k_r50.pth')
        model.load_state_dict(torch.load(weight, map_location=self.device))
        model.eval().to(self.device)
        self.model2 = model

    def size(self):
        return 10

    def generate(self, im_a, im_v, n):
        # att_img
        h, w, c = im_a.shape
        bboxes, kpss = self.detector.detect(im_a, max_num=1)
        if bboxes.shape[0] == 0:
            return None
        att_img, M = norm_crop(im_a, kpss[0], image_size=112)

        # vic_img
        bboxes, kpss = self.detector.detect(im_v, max_num=1)
        if bboxes.shape[0] == 0:
            return None
        vic_img, _ = norm_crop(im_v, kpss[0], image_size=112)

        att_img = att_img[:, :, ::-1]  # BGR ==> RGB
        vic_img = vic_img[:, :, ::-1]

        # get victim feature
        vic_img = torch.Tensor(vic_img.copy()).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
        vic_img.div_(255).sub_(0.5).div_(0.5)  # Normalize
        vic_feat_ir50 = self.model1(vic_img)
        vic_feat_ir100 = self.model2(vic_img)

        # get kpss of att_img afer norm_crop
        bboxes, kpss = self.detector.detect(att_img, max_num=1)
        att_img = att_img.astype(np.uint8)

        # process input
        att_img = torch.Tensor(att_img.copy()).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
        att_img.div_(255).sub_(0.5).div_(0.5)  # Normalize
        att_img_ = att_img.clone()
        att_img.requires_grad = True
        loss_func = nn.CosineEmbeddingLoss()

        # generate mask
        mask = np.zeros((112, 112))
        indices = [0, 1, 2]
        for j in indices:
            x, y = int(kpss[0][j][0]), int(kpss[0][j][1])
            x1 = max(0, x - self.mask_size)
            x2 = min(x + self.mask_size, 112)
            y1 = max(0, y - self.mask_size)
            y2 = min(y + self.mask_size, 112)
            mask[y1:y2, x1:x2] = 1
        mask = torch.from_numpy(mask).to(self.device).float()
        mask_np = cv2.resize(cv2.imread(osp.join('../assets', 'mask_1.png')), (112, 112)) / 255
        mask = torch.Tensor(mask_np.transpose(2, 0, 1)).unsqueeze(0)
        mask = F.interpolate(mask, (112, 112)).to(self.device)
        mask = 1 - mask

        # attack
        g = torch.zeros_like(att_img).detach().to(self.device)
        decay_factor = 0.5
        m = 3
        adv_images = att_img.cpu().clone().detach().to(self.device)
        for iters in tqdm(range(self.num_iters)):
            alpha = self.alphaScheduler.step(iters)
            adv_images.requires_grad = True

            # scale in
            adv_grad = torch.zeros_like(adv_images).detach().to(self.device)
            scale_image = adv_images.clone()
            for i in torch.arange(m):
                scale_image = scale_image / torch.pow(2, i)
                scale_image = scale_image.cpu().detach().to(self.device)
                scale_image.requires_grad = True
                adv_feat_ir50 = self.model1.forward(scale_image)
                adv_feat_ir100 = self.model2.forward(scale_image)

                target = torch.tensor([1]).to(self.device)
                loss_ir50 = loss_func(adv_feat_ir50, vic_feat_ir50, target)
                loss_ir100 = loss_func(adv_feat_ir100, vic_feat_ir100, target)
                loss = loss_ir50 + loss_ir100
                loss.backward(retain_graph=True)
                adv_grad += scale_image.grad.data
            adv_grad /= m

            grad = adv_grad
            g = g * decay_factor + grad / grad.abs().mean(dim=[1, 2, 3], keepdim=True)
            sum_grad = g

            adv_images.data = adv_images.data - torch.sign(sum_grad) * alpha * mask
            adv_images.data = torch.clamp(adv_images.data, -1.0, 1.0)
            att_img.data = adv_images.data

        # get diff
        diff = att_img - att_img_

        # get diff and adv img
        diff = diff.cpu().detach().squeeze().numpy().transpose(1, 2, 0) * 127.5
        diff = cv2.warpAffine(src=diff, M=M, dsize=(w, h), flags=cv2.WARP_INVERSE_MAP, borderValue=0.0)
        diff_bgr = diff[:, :, ::-1]
        adv_img = im_a + diff_bgr
        return adv_img

if __name__ == '__main__':
    if not osp.exists('stage2_output'):
        os.mkdir('stage2_output')
    tool = PyFAT()
    tool.load('../assets')
    num_images = 100
    for i in range(1, 1 + num_images):
        idname = i
        str_idname = "%03d" % idname

        im_a = '../images/'+ str_idname + '/0.png'
        im_a = cv2.imread(im_a)
        im_v = '../images/'+ str_idname + '/1.png'
        im_v = cv2.imread(im_v)
        n = 0
        print('Image-{}/{}'.format(i, n), end="")
        adv_img = tool.generate(im_a, im_v, n)
        save_name = '{}_2.png'.format(str_idname)
        cv2.imwrite('stage2_output/' + save_name, adv_img)