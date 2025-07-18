from __future__ import print_function, division

import argparse
import gc
import os
import time
import random

import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.cuda.amp import autocast as autocast
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import __datasets__
from models import __models__, model_loss_train, model_loss_test
from utils import *

cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(
    description='Fast Stereo Matching through Multi-level Attention Fusion(MAF-Stereo)')
parser.add_argument('--model', default='MAF_Stereo', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')

parser.add_argument('--dataset', default='kitti', help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath_12', default="/dataset/jinlei/Kitti2012/", help='data path')
parser.add_argument('--datapath_15', default="/dataset/jinlei/Kitti2015/", help='data path')
parser.add_argument('--trainlist', default='./filenames/kitti12_15_all.txt', help='training list')
parser.add_argument('--testlist', default='./filenames/kitti15_val.txt', help='testing list')

parser.add_argument('--lr', type=float, default=0.002, help='base learning rate')
parser.add_argument('--batch_size', type=int, default=20, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=16, help='testing batch size')
parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train')
parser.add_argument('--lrepochs', type=str, default="300:10", help='the epochs to decay lr: the downscale rate')

parser.add_argument('--logdir', default='./log/kitti/', help='the directory to save logs and checkpoints')
parser.add_argument('--loadckpt', default='./log/sceneflow.ckpt',
                    help='load the weights from a specific checkpoint')
parser.add_argument('--resume', action='store_true', help='continue training the model')
parser.add_argument('--seed', type=int, default=7, metavar='S', help='random seed')

parser.add_argument('--summary_freq', type=int, default=20, help='the frequency of saving summary')
parser.add_argument('--save_freq', type=int, default=1, help='the frequency of saving checkpoint')
parser.add_argument('--cuda', type=str, default='0', help='cuda number')

# parse arguments, set seeds
args = parser.parse_args()
# cudnn.benchmark = True
cudnn.benchmark = False
cudnn.deterministic = True
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda


# print('epoch {}'.format(args.epoch_num))
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id):
    np.random.seed(args.seed + worker_id)


# torch.manual_seed(args.seed)
# torch.cuda.manual_seed(args.seed)
set_seed(args.seed)
os.makedirs(args.logdir, exist_ok=True)
# create summary logger
print("creating new summary file")
logger = SummaryWriter(args.logdir)

# dataset, dataloader
StereoDataset = __datasets__[args.dataset]
train_dataset = StereoDataset(args.datapath_12, args.datapath_15, args.trainlist, True)
test_dataset = StereoDataset(args.datapath_12, args.datapath_15, args.testlist, False)
TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=8, drop_last=True)
TestImgLoader = DataLoader(test_dataset, args.test_batch_size, shuffle=False, num_workers=4, drop_last=False)

# model, optimizer
model = __models__[args.model](args.maxdisp)
model = nn.DataParallel(model)
model.cuda()
optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr
                                                , epochs=args.epochs,
                                                steps_per_epoch=len(TrainImgLoader)
                                                , anneal_strategy='linear'
                                                )
scaler = torch.cuda.amp.GradScaler()

# load parameters
start_epoch = 0
if args.resume:
    # find all checkpoints file and sort according to epoch id
    all_saved_ckpts = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
    all_saved_ckpts = sorted(all_saved_ckpts, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # use the latest checkpoint file
    loadckpt = os.path.join(args.logdir, all_saved_ckpts[-1])
    print("loading the lastest model in logdir: {}".format(loadckpt))
    state_dict = torch.load(loadckpt)
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    start_epoch = state_dict['epoch'] + 1
elif args.loadckpt:
    # load the checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model_dict = model.state_dict()
    pre_dict = {k: v for k, v in state_dict['model'].items() if k in model_dict}
    model_dict.update(pre_dict)
    model.load_state_dict(model_dict)
print("start at epoch {}".format(start_epoch))


def train():
    bestepoch = 0
    error = 100
    for epoch_idx in range(start_epoch, args.epochs):
        # adjust_learning_rate(optimizer, epoch_idx, args.lr, args.lrepochs)

        # training
        for batch_idx, sample in enumerate(tqdm(TrainImgLoader)):
            if batch_idx == 100:
                break
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            start_time = time.time()
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs = train_sample(sample, compute_metrics=do_summary)
            if do_summary:
                save_scalars(logger, 'train', scalar_outputs, global_step)
                # save_images(logger, 'train', image_outputs, global_step)
            del scalar_outputs
            # print('Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, time = {:.3f}'.format(epoch_idx, args.epochs,
            #                                                                            batch_idx,
            #                                                                            len(TrainImgLoader), loss,
            #                                                                            time.time() - start_time))
        # saving checkpoints
        if (epoch_idx + 1) % args.save_freq == 0:
            checkpoint_data = {'epoch': epoch_idx, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(checkpoint_data, "{}/checkpoint_{:0>6}.ckpt".format(args.logdir, epoch_idx))
        gc.collect()

        # testing
        avg_test_scalars = AverageMeterDict()
        # bestepoch = 0
        # error = 100
        for batch_idx, sample in enumerate(tqdm(TestImgLoader)):
            global_step = len(TestImgLoader) * epoch_idx + batch_idx
            start_time = time.time()
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs = test_sample(sample, compute_metrics=do_summary)
            if do_summary:
                save_scalars(logger, 'test', scalar_outputs, global_step)
                # save_images(logger, 'test', image_outputs, global_step)
            avg_test_scalars.update(scalar_outputs)
            del scalar_outputs
            # print('Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(epoch_idx, args.epochs,
            #                                                                          batch_idx,
            #                                                                          len(TestImgLoader), loss,
            #                                                                          time.time() - start_time))
        avg_test_scalars = avg_test_scalars.mean()
        nowerror = avg_test_scalars["D1"][0]
        if nowerror < error:
            bestepoch = epoch_idx
            error = avg_test_scalars["D1"][0]
        save_scalars(logger, 'fulltest', avg_test_scalars, len(TrainImgLoader) * (epoch_idx + 1))
        print("avg_test_scalars", avg_test_scalars)
        print('MAX epoch %d total test error = %.5f' % (bestepoch, error))
        gc.collect()
    print('MAX epoch %d total test error = %.5f' % (bestepoch, error))


# train one sample
def train_sample(sample, compute_metrics=False):
    model.train()
    imgL, imgR, disp_gt, disp_gt_low = sample['left'], sample['right'], sample['disparity'], sample['disparity_low']
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda()
    disp_gt_low = disp_gt_low.cuda()
    optimizer.zero_grad()

    with autocast():
        disp_ests = model(imgL, imgR)
        mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
        mask_low = (disp_gt_low < args.maxdisp) & (disp_gt_low > 0)
        masks = [mask, mask_low]
        disp_gts = [disp_gt, disp_gt_low]
        loss = model_loss_train(disp_ests, disp_gts, masks)
        disp_ests_final = [disp_ests[0]]

    scalar_outputs = {"loss": loss}
    # image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}
    if compute_metrics:
        with torch.no_grad():
            # image_outputs["errormap"] = [disp_error_image_func()(disp_est, disp_gt) for disp_est in disp_ests_final]
            scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests_final]
            scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests_final]
            # scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests_final]
            # scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests_final]
            # scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests_final]
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()

    return tensor2float(loss), tensor2float(scalar_outputs)


# test one sample
@make_nograd_func
def test_sample(sample, compute_metrics=True):
    model.eval()

    imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disparity']
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda()

    with autocast():
        disp_ests = model(imgL, imgR)
        mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
        masks = [mask]
        disp_gts = [disp_gt]
        loss = model_loss_test(disp_ests, disp_gts, masks)

    scalar_outputs = {"loss": loss}
    # image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}

    scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
    scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
    scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]

    # if compute_metrics:
    #     image_outputs["errormap"] = [disp_error_image_func()(disp_est, disp_gt) for disp_est in disp_ests]

    return tensor2float(loss), tensor2float(scalar_outputs)


if __name__ == '__main__':
    train()
