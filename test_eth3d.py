import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from tqdm import trange

from datasets import ETH3D_loader as et
from datasets.readpfm import readPFM
from models import __models__

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(
    description='Fast Stereo Matching through Multi-level Attention Fusion(MAF-Stereo)')
parser.add_argument('--model', default='MAF_Stereo', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')
parser.add_argument('--datapath', default="/dataset/jinlei/ETH3D/", help='data path')
parser.add_argument('--loadckpt', default='./log/sceneflow.ckpt',
                    help='load the weights from a specific checkpoint')

# parse arguments
args = parser.parse_args()

all_limg, all_rimg, all_disp, all_mask = et.et_loader(args.datapath)

model = __models__[args.model](args.maxdisp)
model = nn.DataParallel(model)
model.cuda()
model.eval()

os.makedirs('./demo/ETH3D/', exist_ok=True)

state_dict = torch.load(args.loadckpt)
model.load_state_dict(state_dict['model'])

pred_mae = 0
pred_op = 0
for i in trange(len(all_limg)):
    limg = Image.open(all_limg[i]).convert('RGB')
    rimg = Image.open(all_rimg[i]).convert('RGB')

    w, h = limg.size
    wi, hi = (w // 32 + 1) * 32, (h // 32 + 1) * 32
    limg = limg.crop((w - wi, h - hi, w, h))
    rimg = rimg.crop((w - wi, h - hi, w, h))

    limg_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(limg)
    rimg_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(rimg)
    limg_tensor = limg_tensor.unsqueeze(0).cuda()
    rimg_tensor = rimg_tensor.unsqueeze(0).cuda()

    disp_gt, _ = readPFM(all_disp[i])
    disp_gt = np.ascontiguousarray(disp_gt, dtype=np.float32)
    disp_gt[disp_gt == np.inf] = 0
    gt_tensor = torch.FloatTensor(disp_gt).unsqueeze(0).unsqueeze(0).cuda()

    occ_mask = np.ascontiguousarray(Image.open(all_mask[i]))

    with torch.no_grad():
        pred_disp = model(limg_tensor, rimg_tensor)[-1]

        pred_disp = pred_disp[:, hi - h:, wi - w:]

    predict_np = pred_disp.squeeze().cpu().numpy()

    op_thresh = 1
    mask = (disp_gt > 0) & (occ_mask == 255)
    # mask = disp_gt > 0
    pred_error = np.abs(predict_np * mask.astype(np.float32) - disp_gt * mask.astype(np.float32))
    print("#######Bad", all_limg[i], np.sum(pred_error > op_thresh) / np.sum(mask))
    pred_op += np.sum(pred_error > op_thresh) / np.sum(mask)
    pred_mae += np.mean(pred_error[mask])

    ########save

    filename = os.path.join('./demo/ETH3D/', all_limg[i].split('/')[-2] + all_limg[i].split('/')[-1])
    pred_np_save = np.round(predict_np * 4 * 256).astype(np.uint16)
    cv2.imwrite(filename, cv2.applyColorMap(cv2.convertScaleAbs(pred_np_save, alpha=0.01), cv2.COLORMAP_JET),
                [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

print("#### EPE", pred_mae / len(all_limg))
print("#### >1.0", pred_op / len(all_limg))
