import os
import os.path as osp
from assets.scrfd import SCRFD
import numpy as np
import random
import torch
import torch.nn.functional as F
import datetime
import cv2
import torch.nn as nn
from tqdm import tqdm
from assets.utils import norm_crop, alphaScheduler, kernelGenerator
from assets import iresnet
import math

class PyFAT:
    def __init__(self, N=10):
        seed = 42
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.alphaScheduler = alphaScheduler(min_alpha=4/255, max_alpha=6/255, T_max=10)
        self.num_iters = 10
        self.device = torch.device('cpu')
        self.mask_size = 9
        self.is_cuda = False

        # Scale Invariant
        self.decay_factor = 0.5
        self.m = 1

        # Diverse Input
        self.resize_rate = 0.9
        self.diversity_prob = 0.8

        self.binomial_prob = 0.9
        
        generator = kernelGenerator('gaussian')
        self.kernel = generator.kernel_generation()
        self.kernel = torch.from_numpy(self.kernel).to(self.device)
        self.len_kernel = generator.len_kernel

    def load(self, assets_path):
        detector = SCRFD(model_file=osp.join(assets_path, 'det_10g.onnx'))
        ctx_id = -1 if not self.is_cuda else 0
        detector.prepare(ctx_id, det_thresh=0.5, input_size=(160, 160))
        self.detector = detector

        model = iresnet.iresnet50()
        weight = osp.join(assets_path, 'w600k_r50.pth')
        model.load_state_dict(torch.load(weight, map_location=self.device))
        model.eval().to(self.device)
        self.model1 = model

        model = iresnet.iresnet100()
        weight = osp.join(assets_path, 'glint360k_r100.pth')
        model.load_state_dict(torch.load(weight, map_location=self.device))
        model.eval().to(self.device)
        self.model2 = model

    def size(self):
        return 10

    def input_diversity(self, x):
        img_size = x.shape[-1]
        img_resize = int(img_size * self.resize_rate)

        if self.resize_rate < 1:
            img_size = img_resize
            img_resize = x.shape[-1]

        rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)

        return padded if torch.rand(1) < self.diversity_prob else x

    def dropout_mask(self):
        mask = np.random.binomial(1, self.binomial_prob, (112, 112))
        mask = torch.from_numpy(mask).to(self.device)
        return mask

    def generate(self, im_a, im_v, n):
        if n != 0:
            im_a = self.im_a

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
        if n == 0:
            self.kpss = kpss
        att_img = att_img.astype(np.uint8)

        # process input
        att_img = torch.Tensor(att_img.copy()).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
        att_img.div_(255).sub_(0.5).div_(0.5)  # Normalize
        att_img_ = att_img.clone()
        att_img.requires_grad = True
        loss_func = nn.CosineEmbeddingLoss()

        # attack
        g = torch.zeros_like(att_img).detach().to(self.device)
        adv_images = att_img.cpu().clone()
        mask = np.zeros((112, 112))
        indices = [0, 1, 2, 3, 4]
        for j in indices:
            x, y = int(self.kpss[0][j][0]), int(self.kpss[0][j][1])
            if j == 2:
                x1 = max(0, x - self.mask_size)
                x2 = min(x + self.mask_size, 112)
                y1 = max(0, y - 5)
                y2 = min(y + 5, 112)
            elif j == 3 or j == 4:
                x1 = max(0, x - self.mask_size)
                x2 = min(x + self.mask_size, 112)
                y1 = max(0, y - self.mask_size + 5)
                y2 = min(y + self.mask_size + 5, 112)
            else:
                x1 = max(0, x - self.mask_size)
                x2 = min(x + self.mask_size, 112)
                y1 = max(0, y - self.mask_size)
                y2 = min(y + self.mask_size, 112)
            mask[y1:y2, x1:x2] = 1
        mask = torch.from_numpy(mask).to(self.device).float()

        adv_images = adv_images.to(self.device)
        for iters in tqdm(range(self.num_iters)):
            alpha = self.alphaScheduler.step(iters)

            # scale in
            adv_grad = torch.zeros_like(adv_images).detach().to(self.device)
            scale_image = adv_images.clone()
            for i in torch.arange(self.m):
                scale_image = scale_image / torch.pow(2, i)
                scale_image = scale_image.cpu().detach().to(self.device)
                scale_image.requires_grad = True
                adv_feat_ir50 = self.model1.forward(self.input_diversity(scale_image))
                adv_feat_ir100 = self.model2.forward(self.input_diversity(scale_image))
                target = torch.tensor([1]).to(self.device)
                loss_ir50 = loss_func(adv_feat_ir50, vic_feat_ir50, target)
                loss_ir100 = loss_func(adv_feat_ir100, vic_feat_ir100, target)
                loss = loss_ir50 + loss_ir100

                loss.backward(retain_graph=True)
                adv_grad += scale_image.grad.data
            adv_grad /= self.m
            adv_grad = F.conv2d(adv_grad.data.clone(), self.kernel, stride=1, padding=(self.len_kernel-1)//2, groups=3)

            grad = adv_grad
            g = g * self.decay_factor + grad / grad.abs().mean(dim=[1, 2, 3], keepdim=True)
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
        self.im_a = adv_img
        return adv_img

if __name__ == '__main__':
    import json
    import base64
    def toBase64Image(x):
        ext = x.split(".")[-1]
        with open(x, "rb") as f:
            img = f.read()
        data = base64.b64encode(img).decode()

        src = "data:image/{ext};base64,{data}".format(ext=ext, data=data)
        return src


    def getScoreOnTencentAPI(im_a, im_v):
        from tencentcloud.common import credential
        from tencentcloud.common.profile.client_profile import ClientProfile
        from tencentcloud.common.profile.http_profile import HttpProfile
        from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
        from tencentcloud.iai.v20200303 import iai_client, models
        try:
            cred = credential.Credential("AKID40hBSVJxPbcQ1OcqtSnHcmiQu9BzZZNw",
                                         "PmrlRxFLK6jGagWBcptlRyJRWQ8mFVYR")
            httpProfile = HttpProfile()
            httpProfile.endpoint = "iai.tencentcloudapi.com"

            clientProfile = ClientProfile()
            clientProfile.httpProfile = httpProfile
            client = iai_client.IaiClient(cred, "ap-chengdu", clientProfile)

            req = models.CompareFaceRequest()
            params = {
                "ImageA": toBase64Image(im_a),
                "ImageB": toBase64Image(im_v),
                "FaceModelVersion": "3.0",
                "QualifyControl": 0
            }
            req.from_json_string(json.dumps(params))

            resp = client.CompareFace(req)
            resp = resp.to_json_string()
            resp = json.loads(resp)

        except TencentCloudSDKException as err:
            print(err)
        return resp['Score']


    def get_connected_domain(img1, img2):
        mask = img1 - img2
        mask[mask != 0] = 255
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours


    def get_score(contours, bbox):
        att_area = 0
        for c in contours:
            rect = cv2.minAreaRect(c)
            att_area = att_area + rect[1][0] * rect[1][1]
        face_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        score = (1 - att_area / face_area) * 100
        return score


    def inference(detector, img):
        bboxes, kpss = detector.detect(img, max_num=1)
        if bboxes.shape[0] == 0:
            return None
        bbox = bboxes[0]
        kp = kpss[0]
        return bbox, kp

    if not osp.exists('stage2_output'):
        os.mkdir('stage2_output')
    tool = PyFAT()
    tool.load('assets')
    num_images = 10
    cnt = 0
    scores = []
    SIM_SCORES = []
    for i in range(1, 1 + num_images):
        idname = i
        str_idname = "%03d" % idname

        im_a = 'casia_image/' + str_idname + '/' + str_idname + '_0.bmp'
        im_a = cv2.imread(im_a)
        im_v = 'casia_image/' + str_idname + '/' + str_idname + '_1.bmp'
        im_v = cv2.imread(im_v)
        sim_scores = []
        for n in range(tool.size()):
            ta = datetime.datetime.now()
            adv_img = tool.generate(im_a, im_v, n)
            tb = datetime.datetime.now()
            print('Image-{} Iter-{} Time: {:.6f}s'.format(i, n + 1, (tb - ta).total_seconds()))
            print('Image-{} Iter-{} Generate time online: {:.6f}s'.format(i, n + 1, (tb - ta).total_seconds() * 4.11))
            save_name = '{}_2.png'.format(str_idname)
            cv2.imwrite('stage2_output/' + save_name, adv_img)

            im_aa = 'stage2_output/' + save_name
            im_vv = 'casia_image/' + str_idname + '/' + str_idname + '_2.bmp'
            sim_score = getScoreOnTencentAPI(im_aa, im_vv)

            hierarchy = get_connected_domain(cv2.imread(im_aa), im_a)
            adv_bbox, _ = inference(tool.detector, cv2.imread(im_aa))
            score = get_score(hierarchy, adv_bbox)
            if sim_score < 50:
                score = 0
            print('Attack Score: ' + str(score))
            sim_scores.append(sim_score)
        for s in sim_scores:
            if s >= 50:
                cnt += 1
                break
        print('Success: {}/{}'.format(cnt, 100))
        SIM_SCORES.append(max(sim_scores))

    for s in SIM_SCORES:
        print(s)