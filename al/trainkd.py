from random import seed
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from collections import OrderedDict
from thop import profile
# from COA_RGBD_SOD.al.models.Segformer_GCN_teacher import RGBD_sal
from COA_SOD.al.models.New_Test_Shunted_T import RGBD_sal
from COA_SOD.al.models.Test_Shunted_S import RGBD_teacher_sal_S
from COA_SOD.al.models.New_Test_Shunted_B import RGBD_teacher_sal
from COA_SOD.al.models.Test_Shunted_B_NOGCN import RGBD_teacher_sal2
from COA_SOD.al.util import Logger, AverageMeter, save_checkpoint, save_tensor_img, set_seed
# from COA_SOD.al.Loss.kd_losses.sp import SP
import os
import numpy as np
from matplotlib import pyplot as plt
import time
import argparse
from tqdm import tqdm
from COA_SOD.al.dataset import get_loader
import torchvision.utils as vutils
import torch.nn.functional as F
# import pytorch_toolbelt.losses as PTL
from COA_SOD.al.config import Config
from COA_SOD.al.loss import saliency_structure_consistency, DSLoss
from COA_SOD.al.util import generate_smoothed_gt
from COA_SOD.al.evaluation.dataloader import EvalDataset
from COA_SOD.al.evaluation.evaluator import Eval_thread
from COA_SOD.al.Loss.losses import KLDLoss, IFVDLoss, ATLoss
# from COA_SOD.al.Loss.kd_losses.ofd import OFD
# from COA_RGBD_SOD.al.Loss.kd_losses.HCL_LOSS import
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# model
# from models GCoNet_plus import GCoNet_plus
# from COA_RGBD_SOD.al.models.GCoNet import GCoNet
# from codes.bayibest82segformerbest.best.newresdecoder4a614t4615622xiuz747117157261015cam11021108110911151116 import M
from COA_SOD.al.pytorch_iou.__init__ import IOU

students_channels = [64, 128, 320, 512]
teachers_channels = [64, 128, 256, 512]

up4 = nn.UpsamplingBilinear2d(scale_factor=4)
up8 = nn.UpsamplingBilinear2d(scale_factor=8)
up16 = nn.UpsamplingBilinear2d(scale_factor=16)
up32 = nn.UpsamplingBilinear2d(scale_factor=32)
conv1_1 = nn.Sequential(nn.Conv2d(students_channels[0], teachers_channels[0], 1, 1, 0)).cuda()
conv2_2 = nn.Sequential(nn.Conv2d(students_channels[1], teachers_channels[1], 1, 1, 0)).cuda()
conv3_3 = nn.Sequential(nn.Conv2d(students_channels[2], teachers_channels[2], 1, 1, 0)).cuda()
conv4_4 = nn.Sequential(nn.Conv2d(students_channels[3], teachers_channels[3], 1, 1, 0)).cuda()
conv_encoder_r = nn.Sequential(nn.Conv2d(students_channels[3], teachers_channels[3], 1, 1, 0)).cuda()
conv_encoder_d = nn.Sequential(nn.Conv2d(students_channels[3], teachers_channels[3], 1, 1, 0)).cuda()
conv_AMGCM = nn.Sequential(nn.Conv2d(students_channels[3], teachers_channels[3], 1, 1, 0)).cuda()
conv_CISA = nn.Sequential(nn.Conv2d(students_channels[2], teachers_channels[2], 1, 1, 0)).cuda()

up2 = nn.UpsamplingBilinear2d(scale_factor=2)
up4 = nn.UpsamplingBilinear2d(scale_factor=4)
up8 = nn.UpsamplingBilinear2d(scale_factor=8)
up16 = nn.UpsamplingBilinear2d(scale_factor=16)
up32 = nn.UpsamplingBilinear2d(scale_factor=32)




def hcl(fstudent, fteacher):
    loss_all = 0.0
    B, C, h, w = fstudent.size()
    loss = F.mse_loss(fstudent, fteacher, reduction='mean')
    cnt = 1.0
    tot = 1.0
    for l in [4,2,1]:
        if l >=h:
            continue
        tmpfs = F.adaptive_avg_pool2d(fstudent, (l,l))
        tmpft = F.adaptive_avg_pool2d(fteacher, (l,l))
        cnt /= 2.0
        loss += F.mse_loss(tmpfs, tmpft, reduction='mean') * cnt
        tot += cnt
    loss = loss / tot
    loss_all = loss_all + loss
    return loss_all

def dice_loss(pred, mask):
    mask = mask
    pred = pred
    intersection = (pred * mask).sum(axis=(2, 3))
    unior = (pred + mask).sum(axis=(2, 3))
    dice = (2 * intersection + 1) / (unior + 1)
    dice = torch.mean(1 - dice)
    return dice


def SP(fm_s, fm_t):
    fm_s = fm_s.view(fm_s.size(0), -1)
    G_s = torch.mm(fm_s, fm_s.t())
    norm_G_s = F.normalize(G_s, p=2, dim=1)

    fm_t = fm_t.view(fm_t.size(0), -1)
    G_t = torch.mm(fm_t, fm_t.t())
    norm_G_t = F.normalize(G_t, p=2, dim=1)

    loss = F.mse_loss(norm_G_s, norm_G_t)

    return loss
# Parameter from command line
parser = argparse.ArgumentParser(description='')
parser.add_argument('--model',
                    default='M',
                    type=str,
                    help="Options: '', ''")
parser.add_argument('--resume',
                    default=None,
                    type=str,
                    help='path to latest checkpoint')
parser.add_argument('--epochs', default=300, type=int)
parser.add_argument('--start_epoch',
                    default=0,
                    type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--trainset',
                    default='DUTS_class',  # default='Jigsaw2_DUTS',
                    type=str,
                    help="Options: 'DUTS_class'")
parser.add_argument('--size',
                    default=256,
                    type=int,
                    help='input size')
parser.add_argument('--ckpt_dir', default="/media/maps/shuju/Alchemist/Shunted_kd/Teacher_New_Test_Shunted_B_Student_Shunted_T_0.8778/Pth",
                    help='Temporary folder')

parser.add_argument('--testsets',
                    default='val5.0',
                    # default='', CoCA+CoSOD3k+Cosal2015  val5.0 RGBD_CoSeg183  RGBD_CoSal1k RGBD_CoSal150
                    type=str,
                    help="Options: 'CoCA','Cosal2015','CoSOD3k','iCoseg','MSRC'")

parser.add_argument('--val_dir',
                    default='tmp4val',
                    type=str,
                    help="Dir for saving tmp results for validation.")

args = parser.parse_args()

config = Config()

# Prepare dataset
if args.trainset == 'DUTS_class':
    root_dir = '/home/maps/Downloads/Alchemist/COA_SOD/Data/'
    train_img_path = os.path.join(root_dir, 'images/DUTS_class')
    train_gt_path = os.path.join(root_dir, 'gts/DUTS_class')
    train_depth_path = os.path.join(root_dir, 'depths/DUTS_class')
    train_edge_path = os.path.join(root_dir, 'edge/DUTS_class')
    train_loader = get_loader(train_img_path,
                              train_gt_path,
                              train_depth_path,
                              train_edge_path,
                              args.size,
                              1,
                              max_num=config.batch_size,
                              istrain=True,
                              shuffle=False,
                              num_workers=8,
                              pin=True)
    train_img_path_seg = os.path.join(root_dir, 'images/coco-seg')
    train_gt_path_seg = os.path.join(root_dir, 'gts/coco-seg')
    train_depth_path_seg = os.path.join(root_dir, 'depths/coco-seg')
    train_edge_path_seg = os.path.join(root_dir, 'gts/coco-seg')
    # train_loader_seg = get_loader(
    #     train_img_path_seg,
    #     train_gt_path_seg,
    #     train_depth_path_seg,
    #     train_edge_path_seg,
    #     args.size,
    #     1,
    #     max_num=config.batch_size,
    #     istrain=True,
    #     shuffle=True,
    #     num_workers=8,
    #     pin=True
    # )
elif args.trainset == 'CoCA':
    root_dir = '/home/maps/Downloads/Alchemist/COA_SOD/Data/'
    train_img_path = os.path.join(root_dir, 'images/CoCA')
    train_gt_path = os.path.join(root_dir, 'gts/CoCA')
    train_depth_path = os.path.join(root_dir, 'depths/CoCA')
    train_loader = get_loader(train_img_path,
                              train_gt_path,
                              train_depth_path,
                              args.size,
                              1,
                              max_num=config.batch_size,
                              istrain=True,
                              shuffle=False,
                              num_workers=8,
                              pin=True)
    train_img_path_seg = os.path.join(root_dir, 'images/coco-seg')
    train_gt_path_seg = os.path.join(root_dir, 'gts/coco-seg')
    train_depth_path_seg = os.path.join(root_dir, 'depths/coco-seg')
    train_loader_seg = get_loader(
        train_img_path_seg,
        train_gt_path_seg,
        train_depth_path_seg,
        args.size,
        1,
        max_num=config.batch_size,
        istrain=True,
        shuffle=True,
        num_workers=8,
        pin=True
    )
else:
    print('Unkonwn train dataset')
    print(args.dataset)

test_loaders = {}
for testset in args.testsets.split('+'):
    test_loader = get_loader(
        os.path.join('/home/maps/Downloads/Alchemist/COA_SOD/Data', 'images', testset),
        os.path.join('/home/maps/Downloads/Alchemist/COA_SOD/Data', 'gts', testset),
        os.path.join('/home/maps/Downloads/Alchemist/COA_SOD/Data', 'depths', testset),
        os.path.join('/home/maps/Downloads/Alchemist/COA_SOD/Data', 'gts', testset),
        args.size, 1, max_num=config.batch_size, istrain=False, shuffle=False, num_workers=8, pin=True
    )
    test_loaders[testset] = test_loader

if config.rand_seed:
    set_seed(config.rand_seed)

# make dir for ckpt
os.makedirs(args.ckpt_dir, exist_ok=True)

# Init log file
logger = Logger(os.path.join(args.ckpt_dir, "log.txt"))
logger_loss_file = os.path.join(args.ckpt_dir, "log_loss.txt")
logger_loss_idx = 1

# Init model
device = torch.device("cuda")
# model_T =RGBD_sal()
model = RGBD_sal()
model_T = RGBD_teacher_sal()
model_T1 = RGBD_teacher_sal2()
model_T2 = RGBD_teacher_sal2()
model_T3 = RGBD_teacher_sal2()
model_T_S = RGBD_teacher_sal_S()
model.load_pre('/home/maps/Downloads/Alchemist/COA_SOD/Shunted/ckpt_T.pth')

base_weights = torch.load('/media/maps/shuju/Alchemist/CoA_pth/Teacher_Pth/New_Shunted_B_ep128_Smeasure0.8778.pth')
new_state_dict = OrderedDict()
for k, v in base_weights.items():
    name = k  # remove 'module.'
    new_state_dict[name] = v

model_T.load_state_dict(new_state_dict)
model_T.eval()

# base_weights1 = torch.load('/media/maps/shuju/Alchemist/CoA_pth/Teacher_Pth/New_Shunted_B_ep128_Smeasure0.8778.pth')
# new_state_dict1 = OrderedDict()
# for k, v in base_weights1.items():
#     name = k  # remove 'module.'
#     new_state_dict1[name] = v
#
# model_T_S.load_state_dict(new_state_dict1)
# model_T_S.eval()

# base_weights = torch.load('/media/maps/shuju/Alchemist/CoA_pth/Teacher_Pth/Shuted_B_NOGCN_ep11_Smeasure0.8714.pth')
# new_state_dict = OrderedDict()
# for k, v in base_weights.items():
#     name = k  # remove 'module.'
#     new_state_dict[name] = v
#
# model_T1.load_state_dict(new_state_dict)
# model_T1.eval()
#
# base_weights = torch.load('/media/maps/shuju/Alchemist/CoA_pth/Teacher_Pth/Shunted_B_NO_GCN_ep37_Smeasure0.8726.pth')
# new_state_dict = OrderedDict()
# for k, v in base_weights.items():
#     name = k  # remove 'module.'
#     new_state_dict[name] = v
#
# model_T2.load_state_dict(new_state_dict)
# model_T2.eval()
#
# base_weights = torch.load('/media/maps/shuju/Alchemist/CoA_pth/Teacher_Pth/Shunted_B_NOGCN_ep29_Smeasure0.8735.pth')
# new_state_dict = OrderedDict()
# for k, v in base_weights.items():
#     name = k  # remove 'module.'
#     new_state_dict[name] = v
#
# model_T3.load_state_dict(new_state_dict)
# model_T3.eval()
# model.load_pre("/home/wby/segformer.b5.640x640.ade.160k.pth")  # 9.28好像忘记加载权重
# model.load_state_dict(torch.load("/media/wby/shuju/ckpt/best_ep15_Smeasure0.8391.pth"),strict=True)
# model.load_pre("/home/wby/Downloads/SegNext/mscan_l.pth")
model_T_S = model_T_S.to(device)
model_T = model_T.to(device)
model_T1 = model_T1.to(device)
model_T2 = model_T2.to(device)
model_T3 = model_T3.to(device)
model = model.to(device)
if config.lambda_adv:
    from COA_SOD.al.adv import Discriminator

    disc = Discriminator(channels=1, img_size=args.size).to(device)
    optimizer_d = optim.Adam(params=disc.parameters(), lr=config.lr, betas=[0.9, 0.99])
    Tensor = torch.cuda.FloatTensor if (True if torch.cuda.is_available() else False) else torch.FloatTensor
    adv_criterion = nn.BCELoss()

CE = torch.nn.BCEWithLogitsLoss().cuda()
IOU = IOU(size_average=True).cuda()
KLDLoss1 = KLDLoss().cuda()
KLDLoss2 = KLDLoss().cuda()
KLDLoss3 = KLDLoss().cuda()
KLDLoss4 = KLDLoss().cuda()
KLDLoss5 = KLDLoss().cuda()
KLDLoss6 = KLDLoss().cuda()
KLDLoss7 = KLDLoss().cuda()
KLDLoss8 = KLDLoss().cuda()
KLDLoss9 = KLDLoss().cuda()
KLDLoss10 = KLDLoss().cuda()
# sploss = SP().cuda()
AT_Loss = ATLoss().cuda()

IFVDLoss = IFVDLoss().cuda()
# dice = dice_loss().cuda()
# OFDLoss1 = OFD(in_channels=64, out_channels=64).cuda()
# OFDLoss2 = OFD(in_channels=128, out_channels=128).cuda()
# OFDLoss3 = OFD(in_channels=256, out_channels=256).cuda()
# OFDLoss4 = OFD(in_channels=512, out_channels=512).cuda()

# HCLoss1 = hcl().cuda()
# HCLoss2 = hcl().cuda()
# HCLoss3 = hcl().cuda()
# HCLoss4 = hcl().cuda()
# HCLossr = hcl().cuda()
# HCLossd = hcl().cuda()
# HCLossA = hcl().cuda()
# HCLossA1 = hcl().cuda()
# HCLossA1 = hcl().cuda()
# HCLossA2 = hcl().cuda()
# HCLossA3 = hcl().cuda()
# HCLossA4 = hcl().cuda()



all_params = model.parameters()
# Setting optimizer
optimizer = optim.Adam(params=all_params, lr=config.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.decay_step_size, gamma=0.1)

# Why freeze the backbone?...
if config.freeze:
    for key, value in model.named_parameters():
        if 'bb' in key and 'bb.conv5.conv5_3' not in key:
            value.requires_grad = False

# log model and optimizer params
logger.info("Model details:")
logger.info(model)
logger.info("Optimizer details:")
logger.info(optimizer)
logger.info("Scheduler details:")
logger.info(scheduler)
logger.info("Other hyperparameters:")
logger.info(args)

# Setting Loss
dsloss = DSLoss()


def main():
    val_measures = []
    global temperature
    temperature = 34
    # Optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            model.load_state_dict(torch.load(args.resume))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    for epoch in range(args.start_epoch, args.epochs):
        train_loss = train(epoch, temperature)
        print('temperature:', temperature)
        if temperature != 1:
            temperature -= 3
            print('Change temperature to:', str(temperature))
        if config.validation:
            measures = validate(model, test_loaders, args.testsets, temperature)
            val_measures.append(measures)
            print(
                'Validation: S_measure on CoCA for epoch-{} is {:.4f}. Best epoch is epoch-{} with S_measure {:.4f}'.format(
                    epoch, measures[0], np.argmax(np.array(val_measures)[:, 0].squeeze()),
                    np.max(np.array(val_measures)[:, 0]))
            )
            # print(
            #     'Validation: MAE on CoCA for epoch-{} is {:.4f}. Best epoch is epoch-{} with MAE {:.4f}'.format(
            #         epoch, measures[0], np.argmax(np.array(val_measures)[:, 0].squeeze()),
            #         np.max(np.array(val_measures)[:, 0]))
            # )
        # Save checkpoint
        save_checkpoint(
            {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'scheduler': scheduler.state_dict(),
            },
            path=args.ckpt_dir)
        # if epoch >= args.epochs - config.val_last:

        # if epoch >= args.epochs - 50:
        if epoch >= args.epochs - 5 or measures[0] >= 0.8600:
            torch.save(model.state_dict(), os.path.join(args.ckpt_dir, 'Segformer_GCN_ep{}_Smeasure{:.4f}.pth'.format(epoch, measures[0])))

        if config.validation:
            if np.max(np.array(val_measures)[:, 0].squeeze()) == measures[0]:
                best_weights_before = [os.path.join(args.ckpt_dir, weight_file) for weight_file in
                                       os.listdir(args.ckpt_dir) if 'best_' in weight_file]
                for best_weight_before in best_weights_before:
                    os.remove(best_weight_before)
                torch.save(model.state_dict(), os.path.join(args.ckpt_dir,'Segformer_GCN_best_ep{}_Smeasure{:.4f}.pth'.format(epoch, measures[0])))




def train(epoch, temperature):
    loss_log = AverageMeter()
    loss_log_triplet = AverageMeter()
    global logger_loss_idx
    model.train()
    model_T.eval()

    # for batch_idx, (batch, batch_seg) in enumerate(zip(train_loader, train_loader_seg)):
    for batch_idx, batch in enumerate(train_loader):
        inputs = batch[0].to(device).squeeze(0)
        gts = batch[1].to(device).squeeze(0)
        depths = batch[2].to(device).squeeze(0)
        back = 1 - gts
        # edges = batch[3].to(device).squeeze(0)
        # cls_gts = torch.LongTensor(batch[-1]).to(device)
        # print("gt--depth",gts.size(),edges.size())
        # print(cls_gts)


        # with torch.no_grad():
        #     T_S_1, T_S_2, T_S_3, T_S_4, T_F_1, T_F_2, T_F_3, T_F_4, T_R_4, \
        #         T_D_4, T_r_1, T_r_2, T_r_3, T_r_4, T_d_1, T_d_2, T_d_3, T_d_4 = model_T(inputs, depths)
        #
        #     T1_S_1, T1_S_2, T1_S_3, T1_S_4, T1_F_1, T1_F_2, T1_F_3, T1_F_4, T1_R_4, \
        #         T1_D_4, T1_r_1, T1_r_2, T1_r_3, T1_r_4, T1_d_1, T1_d_2, T1_d_3, T1_d_4 = model_T_S(inputs, depths)

        # false_labels = model_T(inputs, depths)
        # S_S_1, S_S_2, S_S_3, S_S_4, S_F_1, S_F_2, S_F_3, S_F_4, S_R_4, \
        #     S_D_4, S_r_1, S_r_2, S_r_3, S_r_4, S_d_1, S_d_2, S_d_3, S_d_4 = model(inputs, depths)
        # gts = false_labels[-1]
        # print(false_labels[0].shape)
        # print(return_values[0].shape)
        # print(gts.shape)
        # Loss
        # cls_percentage = F.adaptive_avg_pool2d(gts, 4)
        loss = 0

        # if epoch <= 20:
        #     TS_S_1, TS_S_2, TS_S_3, TS_S_4, TS_F_1, TS_F_2, TS_F_3, TS_F_4, TS_R_4, \
        #         TS_D_4, TS_r_1, TS_r_2, TS_r_3, TS_r_4, TS_d_1, TS_d_2, TS_d_3, TS_d_4 = model_T_S(inputs, depths)
        #
        #     s_s_1 = F.sigmoid(S_S_1)
        #     s_s_2 = F.sigmoid(S_S_2)
        #     s_s_3 = F.sigmoid(S_S_3)
        #     s_s_4 = F.sigmoid(S_S_4)
        #
        #     ts_s_1 = F.sigmoid(TS_S_1)
        #     ts_s_2 = F.sigmoid(TS_S_2)
        #     ts_s_3 = F.sigmoid(TS_S_3)
        #     ts_s_4 = F.sigmoid(TS_S_4)
        #
        #     s_s_1 = 1 - s_s_1
        #     s_s_2 = 1 - s_s_2
        #     s_s_3 = 1 - s_s_3
        #     s_s_4 = 1 - s_s_4
        #
        #     ts_s_1 = 1 - ts_s_1
        #     ts_s_2 = 1 - ts_s_2
        #     ts_s_3 = 1 - ts_s_3
        #     ts_s_4 = 1 - ts_s_4
        #
        #     TS_loss_encoder_R_4 = SP(S_R_4 / temperature, TS_R_4 / temperature).cuda() * 4
        #     TS_loss_encoder_D_4 = SP(S_D_4 / temperature, TS_D_4 / temperature).cuda() * 4
        #
        #     TS_loss_decoder_r_1 = SP(S_r_1 / temperature, TS_r_1 / temperature).cuda()
        #     TS_loss_decoder_r_2 = SP(S_r_2 / temperature, TS_r_2 / temperature).cuda()
        #     TS_loss_decoder_r_3 = SP(S_r_3 / temperature, TS_r_3 / temperature).cuda()
        #     TS_loss_decoder_r_4 = SP(S_r_4 / temperature, TS_r_4 / temperature).cuda()
        #     TS_loss_decoder_d_1 = SP(S_d_1 / temperature, TS_d_1 / temperature).cuda()
        #     TS_loss_decoder_d_2 = SP(S_d_2 / temperature, TS_d_2 / temperature).cuda()
        #     TS_loss_decoder_d_3 = SP(S_d_3 / temperature, TS_d_3 / temperature).cuda()
        #     TS_loss_decoder_d_4 = SP(S_d_4 / temperature, TS_d_4 / temperature).cuda()
        #
        #     TS_loss_encoder = TS_loss_encoder_R_4 + TS_loss_encoder_D_4 + \
        #                       TS_loss_decoder_r_1 + TS_loss_decoder_r_2 + TS_loss_decoder_r_3 + TS_loss_decoder_r_4 + \
        #                       TS_loss_decoder_d_1 + TS_loss_decoder_d_2 + TS_loss_decoder_d_3 + TS_loss_decoder_d_4
        #
        #     loss_encoder = TS_loss_encoder
        #
        #     TS_loss_F_1 = AT_Loss(S_F_1 / temperature, TS_F_1 / temperature, gts, 4)
        #     TS_loss_F_2 = AT_Loss(S_F_2 / temperature, TS_F_2 / temperature, gts, 4)
        #     TS_loss_F_3 = AT_Loss(S_F_3 / temperature, TS_F_3 / temperature, gts, 4)
        #     TS_loss_F_4 = AT_Loss(S_F_4 / temperature, TS_F_4 / temperature, gts, 4)
        #
        #     TS_loss_F = TS_loss_F_1 * 4 + TS_loss_F_2 + TS_loss_F_3 + TS_loss_F_4
        #     loss_F = TS_loss_F
        #
        #     loss_gt1 = CE(S_S_1, gts)
        #     loss_gt2 = CE(S_S_2, gts)
        #     loss_gt3 = CE(S_S_3, gts)
        #     loss_gt4 = CE(S_S_4, gts)
        #     loss_gt = loss_gt1 * 4 + loss_gt2 + loss_gt3 + loss_gt4
        #
        #     TS_loss_kd_1_b = KLDLoss1(s_s_1 / temperature, ts_s_1 / temperature, gts, 4).cuda()
        #     TS_loss_kd_2_b = KLDLoss2(s_s_2 / temperature, ts_s_2 / temperature, gts, 4).cuda()
        #     TS_loss_kd_3_b = KLDLoss3(s_s_3 / temperature, ts_s_3 / temperature, gts, 4).cuda()
        #     TS_loss_kd_4_b = KLDLoss4(s_s_4 / temperature, ts_s_4 / temperature, gts, 4).cuda()
        #     TS_loss_kd_b = TS_loss_kd_1_b * 4 + TS_loss_kd_2_b + TS_loss_kd_3_b + TS_loss_kd_4_b
        #
        #     loss_kd_b = TS_loss_kd_b
        #
        #     TS_loss_kd_1 = KLDLoss1(S_S_1 / 4, TS_S_1 / 4, gts, 4) * 4
        #     TS_loss_kd_2 = KLDLoss2(S_S_2 / 4, TS_S_2 / 4, gts, 4)
        #     TS_loss_kd_3 = KLDLoss3(S_S_3 / 4, TS_S_3 / 4, gts, 4)
        #     TS_loss_kd_4 = KLDLoss4(S_S_4 / 4, TS_S_4 / 4, gts, 4)
        #     TS_loss_kd = (TS_loss_kd_1 + TS_loss_kd_2 + TS_loss_kd_3 + TS_loss_kd_4) * (1 - 0.0625) + 0.0625 * loss_gt
        #
        #     loss_kd = TS_loss_kd
        #
        # elif 20 < epoch <= 40:
        #
        #     with torch.no_grad():
        #         TS_S_1, TS_S_2, TS_S_3, TS_S_4, TS_F_1, TS_F_2, TS_F_3, TS_F_4, TS_R_4,\
        #             TS_D_4, TS_r_1, TS_r_2, TS_r_3, TS_r_4, TS_d_1, TS_d_2, TS_d_3, TS_d_4 = model_T_S(inputs, depths)
        #
        #         T1_S_1, T1_S_2, T1_S_3, T1_S_4, T1_F_1, T1_F_2, T1_F_3, T1_F_4, T1_R_4, \
        #             T1_D_4, T1_r_1, T1_r_2, T1_r_3, T1_r_4, T1_d_1, T1_d_2, T1_d_3, T1_d_4 = model_T1(inputs, depths)
        #         s_s_1 = F.sigmoid(S_S_1)
        #         s_s_2 = F.sigmoid(S_S_2)
        #         s_s_3 = F.sigmoid(S_S_3)
        #         s_s_4 = F.sigmoid(S_S_4)
        #
        #         ts_s_1 = F.sigmoid(TS_S_1)
        #         ts_s_2 = F.sigmoid(TS_S_2)
        #         ts_s_3 = F.sigmoid(TS_S_3)
        #         ts_s_4 = F.sigmoid(TS_S_4)
        #
        #         t1_s_1 = F.sigmoid(T1_S_1)
        #         t1_s_2 = F.sigmoid(T1_S_2)
        #         t1_s_3 = F.sigmoid(T1_S_3)
        #         t1_s_4 = F.sigmoid(T1_S_4)
        #
        #         s_s_1 = 1 - s_s_1
        #         s_s_2 = 1 - s_s_2
        #         s_s_3 = 1 - s_s_3
        #         s_s_4 = 1 - s_s_4
        #
        #         ts_s_1 = 1 - ts_s_1
        #         ts_s_2 = 1 - ts_s_2
        #         ts_s_3 = 1 - ts_s_3
        #         ts_s_4 = 1 - ts_s_4
        #
        #         t1_s_1 = 1 - t1_s_1
        #         t1_s_2 = 1 - t1_s_2
        #         t1_s_3 = 1 - t1_s_3
        #         t1_s_4 = 1 - t1_s_4
        #
        #     TS_loss_encoder_R_4 = SP(S_R_4 / temperature, TS_R_4 / temperature).cuda() * 4
        #     TS_loss_encoder_D_4 = SP(S_D_4 / temperature, TS_D_4 / temperature).cuda() * 4
        #
        #     TS_loss_decoder_r_1 = SP(S_r_1 / temperature, TS_r_1 / temperature).cuda()
        #     TS_loss_decoder_r_2 = SP(S_r_2 / temperature, TS_r_2 / temperature).cuda()
        #     TS_loss_decoder_r_3 = SP(S_r_3 / temperature, TS_r_3 / temperature).cuda()
        #     TS_loss_decoder_r_4 = SP(S_r_4 / temperature, TS_r_4 / temperature).cuda()
        #     TS_loss_decoder_d_1 = SP(S_d_1 / temperature, TS_d_1 / temperature).cuda()
        #     TS_loss_decoder_d_2 = SP(S_d_2 / temperature, TS_d_2 / temperature).cuda()
        #     TS_loss_decoder_d_3 = SP(S_d_3 / temperature, TS_d_3 / temperature).cuda()
        #     TS_loss_decoder_d_4 = SP(S_d_4 / temperature, TS_d_4 / temperature).cuda()
        #
        #     T1_loss_encoder_R_4 = SP(S_R_4 / 4, T1_R_4 / 4).cuda() * 4
        #     T1_loss_encoder_D_4 = SP(S_D_4 / 4, T1_D_4 / 4).cuda() * 4
        #
        #     T1_loss_decoder_r_1 = SP(S_r_1 / temperature, T1_r_1 / temperature).cuda()
        #     T1_loss_decoder_r_2 = SP(S_r_2 / temperature, T1_r_2 / temperature).cuda()
        #     T1_loss_decoder_r_3 = SP(S_r_3 / temperature, T1_r_3 / temperature).cuda()
        #     T1_loss_decoder_r_4 = SP(S_r_4 / temperature, T1_r_4 / temperature).cuda()
        #     T1_loss_decoder_d_1 = SP(S_d_1 / temperature, T1_d_1 / temperature).cuda()
        #     T1_loss_decoder_d_2 = SP(S_d_2 / temperature, T1_d_2 / temperature).cuda()
        #     T1_loss_decoder_d_3 = SP(S_d_3 / temperature, T1_d_3 / temperature).cuda()
        #     T1_loss_decoder_d_4 = SP(S_d_4 / temperature, T1_d_4 / temperature).cuda()
        #
        #     TS_loss_encoder = TS_loss_encoder_R_4 + TS_loss_encoder_D_4 + \
        #                      TS_loss_decoder_r_1 + TS_loss_decoder_r_2 + TS_loss_decoder_r_3 + TS_loss_decoder_r_4 + \
        #                      TS_loss_decoder_d_1 + TS_loss_decoder_d_2 + TS_loss_decoder_d_3 + TS_loss_decoder_d_4
        #
        #     T1_loss_encoder = T1_loss_encoder_R_4 + T1_loss_encoder_D_4 + \
        #                       T1_loss_decoder_r_1 + T1_loss_decoder_r_2 + T1_loss_decoder_r_3 + T1_loss_decoder_r_4 + \
        #                       T1_loss_decoder_d_1 + T1_loss_decoder_d_2 + T1_loss_decoder_d_3 + T1_loss_decoder_d_4
        #     loss_encoder = TS_loss_encoder + T1_loss_encoder / 2
        #
        #     TS_loss_F_1 = AT_Loss(S_F_1 / temperature, TS_F_1 / temperature, gts, 4)
        #     TS_loss_F_2 = AT_Loss(S_F_2 / temperature, TS_F_2 / temperature, gts, 4)
        #     TS_loss_F_3 = AT_Loss(S_F_3 / temperature, TS_F_3 / temperature, gts, 4)
        #     TS_loss_F_4 = AT_Loss(S_F_4 / temperature, TS_F_4 / temperature, gts, 4)
        #
        #     TS_loss_F = TS_loss_F_1 * 4 + TS_loss_F_2 + TS_loss_F_3 + TS_loss_F_4
        #
        #     T1_loss_F_1 = AT_Loss(S_F_1 / temperature, T1_F_1 / temperature, gts, 4)
        #     T1_loss_F_2 = AT_Loss(S_F_2 / temperature, T1_F_2 / temperature, gts, 4)
        #     T1_loss_F_3 = AT_Loss(S_F_3 / temperature, T1_F_3 / temperature, gts, 4)
        #     T1_loss_F_4 = AT_Loss(S_F_4 / temperature, T1_F_4 / temperature, gts, 4)
        #
        #     T1_loss_F = T1_loss_F_1 * 4 + T1_loss_F_2 + T1_loss_F_3 + T1_loss_F_4
        #
        #     loss_F = TS_loss_F + T1_loss_F / 2
        #
        #     loss_gt1 = CE(S_S_1, gts)
        #     loss_gt2 = CE(S_S_2, gts)
        #     loss_gt3 = CE(S_S_3, gts)
        #     loss_gt4 = CE(S_S_4, gts)
        #     loss_gt = loss_gt1 * 4 + loss_gt2 + loss_gt3 + loss_gt4
        #
        #     TS_loss_kd_1_b = KLDLoss1(s_s_1 / temperature, ts_s_1 / temperature, gts, 4).cuda()
        #     TS_loss_kd_2_b = KLDLoss2(s_s_2 / temperature, ts_s_2 / temperature, gts, 4).cuda()
        #     TS_loss_kd_3_b = KLDLoss3(s_s_3 / temperature, ts_s_3 / temperature, gts, 4).cuda()
        #     TS_loss_kd_4_b = KLDLoss4(s_s_4 / temperature, ts_s_4 / temperature, gts, 4).cuda()
        #     TS_loss_kd_b = TS_loss_kd_1_b * 4 + TS_loss_kd_2_b + TS_loss_kd_3_b + TS_loss_kd_4_b
        #
        #     T1_loss_kd_1_b = KLDLoss1(s_s_1 / temperature, t1_s_1 / temperature, gts, 4).cuda()
        #     T1_loss_kd_2_b = KLDLoss2(s_s_2 / temperature, t1_s_2 / temperature, gts, 4).cuda()
        #     T1_loss_kd_3_b = KLDLoss3(s_s_3 / temperature, t1_s_3 / temperature, gts, 4).cuda()
        #     T1_loss_kd_4_b = KLDLoss4(s_s_4 / temperature, t1_s_4 / temperature, gts, 4).cuda()
        #     T1_loss_kd_b = T1_loss_kd_1_b * 4 + T1_loss_kd_2_b + T1_loss_kd_3_b + T1_loss_kd_4_b
        #
        #     loss_kd_b = TS_loss_kd_b + T1_loss_kd_b / 2
        #
        #     TS_loss_kd_1 = KLDLoss1(S_S_1 / 4, TS_S_1 / 4, gts, 4) * 4
        #     TS_loss_kd_2 = KLDLoss2(S_S_2 / 4, TS_S_2 / 4, gts, 4)
        #     TS_loss_kd_3 = KLDLoss3(S_S_3 / 4, TS_S_3 / 4, gts, 4)
        #     TS_loss_kd_4 = KLDLoss4(S_S_4 / 4, TS_S_4 / 4, gts, 4)
        #     TS_loss_kd = (TS_loss_kd_1 + TS_loss_kd_2 + TS_loss_kd_3 + TS_loss_kd_4) * (1 - 0.0625) + 0.0625 * loss_gt
        #
        #     T1_loss_kd_1 = KLDLoss1(S_S_1 / temperature, T1_S_1 / temperature, gts, 4) * 4
        #     T1_loss_kd_2 = KLDLoss2(S_S_2 / temperature, T1_S_2 / temperature, gts, 4)
        #     T1_loss_kd_3 = KLDLoss3(S_S_3 / temperature, T1_S_3 / temperature, gts, 4)
        #     T1_loss_kd_4 = KLDLoss4(S_S_4 / temperature, T1_S_4 / temperature, gts, 4)
        #     T1_loss_kd = (T1_loss_kd_1 + T1_loss_kd_2 + T1_loss_kd_3 + T1_loss_kd_4) * (1 - 0.0625) + 0.0625 * loss_gt
        #
        #     loss_kd = TS_loss_kd + T1_loss_kd / 2
        #
        #
        # elif 40 < epoch <= 70:
        #
        #     with torch.no_grad():
        #         TS_S_1, TS_S_2, TS_S_3, TS_S_4, TS_F_1, TS_F_2, TS_F_3, TS_F_4, TS_R_4, \
        #             TS_D_4, TS_r_1, TS_r_2, TS_r_3, TS_r_4, TS_d_1, TS_d_2, TS_d_3, TS_d_4 = model_T_S(inputs, depths)
        #
        #         T1_S_1, T1_S_2, T1_S_3, T1_S_4, T1_F_1, T1_F_2, T1_F_3, T1_F_4, T1_R_4, \
        #             T1_D_4, T1_r_1, T1_r_2, T1_r_3, T1_r_4, T1_d_1, T1_d_2, T1_d_3, T1_d_4 = model_T1(inputs, depths)
        #
        #         T2_S_1, T2_S_2, T2_S_3, T2_S_4, T2_F_1, T2_F_2, T2_F_3, T2_F_4, T2_R_4, \
        #             T2_D_4, T2_r_1, T2_r_2, T2_r_3, T2_r_4, T2_d_1, T2_d_2, T2_d_3, T2_d_4 = model_T2(inputs, depths)
        #
        #     TS_loss_encoder_R_4 = SP(S_R_4 / 4, TS_R_4 / 4).cuda() * 4
        #     TS_loss_encoder_D_4 = SP(S_D_4 / 4, TS_D_4 / 4).cuda() * 4
        #
        #     T1_loss_encoder_R_4 = SP(S_R_4 / 4, T1_R_4 / 4).cuda() * 4
        #     T1_loss_encoder_D_4 = SP(S_D_4 / 4, T1_D_4 / 4).cuda() * 4
        #
        #     T2_loss_encoder_R_4 = SP(S_R_4 / 4, T2_R_4 / 4).cuda() * 4
        #     T2_loss_encoder_D_4 = SP(S_D_4 / 4, T2_D_4 / 4).cuda() * 4
        #
        #     TS_loss_encoder = TS_loss_encoder_R_4 + TS_loss_encoder_D_4
        #     T1_loss_encoder = T1_loss_encoder_R_4 + T1_loss_encoder_D_4
        #     T2_loss_encoder = T2_loss_encoder_R_4 + T2_loss_encoder_D_4
        #
        #     loss_encoder = TS_loss_encoder + T1_loss_encoder + T2_loss_encoder / 2
        #
        #     TS_loss_F_1 = AT_Loss(S_F_1 / temperature, TS_F_1 / temperature, gts, 4)
        #     TS_loss_F_2 = AT_Loss(S_F_2 / temperature, TS_F_2 / temperature, gts, 4)
        #     TS_loss_F_3 = AT_Loss(S_F_3 / temperature, TS_F_3 / temperature, gts, 4)
        #     TS_loss_F_4 = AT_Loss(S_F_4 / temperature, TS_F_4 / temperature, gts, 4)
        #
        #     TS_loss_F = TS_loss_F_1 * 4 + TS_loss_F_2 + TS_loss_F_3 + TS_loss_F_4
        #
        #     T1_loss_F_1 = AT_Loss(S_F_1 / 4, T1_F_1 / 4, gts, 4)
        #     T1_loss_F_2 = AT_Loss(S_F_2 / 4, T1_F_2 / 4, gts, 4)
        #     T1_loss_F_3 = AT_Loss(S_F_3 / 4, T1_F_3 / 4, gts, 4)
        #     T1_loss_F_4 = AT_Loss(S_F_4 / 4, T1_F_4 / 4, gts, 4)
        #
        #     T1_loss_F = T1_loss_F_1 * 4 + T1_loss_F_2 + T1_loss_F_3 + T1_loss_F_4
        #
        #     T2_loss_F_1 = AT_Loss(S_F_1 / 4, T2_F_1 / 4, gts, 4)
        #     T2_loss_F_2 = AT_Loss(S_F_2 / 4, T2_F_2 / 4, gts, 4)
        #     T2_loss_F_3 = AT_Loss(S_F_3 / 4, T2_F_3 / 4, gts, 4)
        #     T2_loss_F_4 = AT_Loss(S_F_4 / 4, T2_F_4 / 4, gts, 4)
        #
        #     T2_loss_F = T2_loss_F_1 * 4 + T2_loss_F_2 + T2_loss_F_3 + T2_loss_F_4
        #
        #     loss_F = TS_loss_F + T1_loss_F + T2_loss_F
        #
        #     loss_gt1 = CE(S_S_1, gts)
        #     loss_gt2 = CE(S_S_2, gts)
        #     loss_gt = loss_gt1 * 4 + loss_gt2
        #     loss_kd_b = 0
        #
        #     TS_loss_kd_1 = KLDLoss1(S_S_1 / temperature, TS_S_1 / temperature, gts, 4) * 4
        #     TS_loss_kd_2 = KLDLoss2(S_S_2 / temperature, TS_S_2 / temperature, gts, 4)
        #     TS_loss_kd_3 = KLDLoss3(S_S_3 / temperature, TS_S_3 / temperature, gts, 4)
        #     TS_loss_kd_4 = KLDLoss4(S_S_4 / temperature, TS_S_4 / temperature, gts, 4)
        #     TS_loss_kd = (TS_loss_kd_1 + TS_loss_kd_2 + TS_loss_kd_3 + TS_loss_kd_4) * (1 - 0.0625) + 0.0625 * loss_gt
        #
        #     T1_loss_kd_1 = KLDLoss1(S_S_1 / temperature, T1_S_1 / temperature, gts, 4) * 4
        #     T1_loss_kd_2 = KLDLoss2(S_S_2 / temperature, T1_S_2 / temperature, gts, 4)
        #     T1_loss_kd_3 = KLDLoss3(S_S_3 / temperature, T1_S_3 / temperature, gts, 4)
        #     T1_loss_kd_4 = KLDLoss4(S_S_4 / temperature, T1_S_4 / temperature, gts, 4)
        #     T1_loss_kd = (T1_loss_kd_1 + T1_loss_kd_2 + T1_loss_kd_3 + T1_loss_kd_4) * (1 - 0.0625) + 0.0625 * loss_gt
        #
        #     T2_loss_kd_1 = KLDLoss1(S_S_1 / 4, T2_S_1 / 4, gts, 4) * 4
        #     T2_loss_kd_2 = KLDLoss2(S_S_2 / 4, T2_S_2 / 4, gts, 4)
        #     T2_loss_kd_3 = KLDLoss3(S_S_3 / 4, T2_S_3 / 4, gts, 4)
        #     T2_loss_kd_4 = KLDLoss4(S_S_4 / 4, T2_S_4 / 4, gts, 4)
        #     T2_loss_kd = (T2_loss_kd_1 + T2_loss_kd_2 + T2_loss_kd_3 + T2_loss_kd_4) * (1 - 0.0625) + 0.0625 * loss_gt
        #
        #     loss_kd = TS_loss_kd + T1_loss_kd + T2_loss_kd
        #
        #
        # elif 70< epoch<=100:
        #     with torch.no_grad():
        #         TS_S_1, TS_S_2, TS_S_3, TS_S_4, TS_F_1, TS_F_2, TS_F_3, TS_F_4, TS_R_4, \
        #             TS_D_4, TS_r_1, TS_r_2, TS_r_3, TS_r_4, TS_d_1, TS_d_2, TS_d_3, TS_d_4 = model_T_S(inputs, depths)
        #
        #         T1_S_1, T1_S_2, T1_S_3, T1_S_4, T1_F_1, T1_F_2, T1_F_3, T1_F_4, T1_R_4, \
        #             T1_D_4, T1_r_1, T1_r_2, T1_r_3, T1_r_4, T1_d_1, T1_d_2, T1_d_3, T1_d_4 = model_T1(inputs, depths)
        #
        #         T2_S_1, T2_S_2, T2_S_3, T2_S_4, T2_F_1, T2_F_2, T2_F_3, T2_F_4, T2_R_4, \
        #             T2_D_4, T2_r_1, T2_r_2, T2_r_3, T2_r_4, T2_d_1, T2_d_2, T2_d_3, T2_d_4 = model_T2(inputs, depths)
        #
        #         T3_S_1, T3_S_2, T3_S_3, T3_S_4, T3_F_1, T3_F_2, T3_F_3, T3_F_4, T3_R_4, \
        #             T3_D_4, T3_r_1, T3_r_2, T3_r_3, T3_r_4, T3_d_1, T3_d_2, T3_d_3, T3_d_4 = model_T3(inputs, depths)
        #
        #     TS_loss_encoder_R_4 = SP(S_R_4 / 4, TS_R_4 / 4).cuda() * 4
        #     TS_loss_encoder_D_4 = SP(S_D_4 / 4, TS_D_4 / 4).cuda() * 4
        #
        #     T1_loss_encoder_R_4 = SP(S_R_4 / 4, T1_R_4 / 4).cuda() * 4
        #     T1_loss_encoder_D_4 = SP(S_D_4 / 4, T1_D_4 / 4).cuda() * 4
        #
        #     T2_loss_encoder_R_4 = SP(S_R_4 / 4, T2_R_4 / 4).cuda() * 4
        #     T2_loss_encoder_D_4 = SP(S_D_4 / 4, T2_D_4 / 4).cuda() * 4
        #
        #     T3_loss_encoder_R_4 = SP(S_R_4 / 4, T3_R_4 / 4).cuda() * 4
        #     T3_loss_encoder_D_4 = SP(S_D_4 / 4, T3_D_4 / 4).cuda() * 4
        #
        #     TS_loss_encoder = TS_loss_encoder_R_4 + TS_loss_encoder_D_4
        #     T1_loss_encoder = T1_loss_encoder_R_4 + T1_loss_encoder_D_4
        #     T2_loss_encoder = T2_loss_encoder_R_4 + T2_loss_encoder_D_4
        #     T3_loss_encoder = T3_loss_encoder_R_4 + T3_loss_encoder_D_4
        #
        #
        #     loss_encoder = TS_loss_encoder + T1_loss_encoder + T2_loss_encoder + T3_loss_encoder / 2
        #
        #     TS_loss_F_1 = AT_Loss(S_F_1 / temperature, TS_F_1 / temperature, gts, 4)
        #     TS_loss_F_2 = AT_Loss(S_F_2 / temperature, TS_F_2 / temperature, gts, 4)
        #     TS_loss_F_3 = AT_Loss(S_F_3 / temperature, TS_F_3 / temperature, gts, 4)
        #     TS_loss_F_4 = AT_Loss(S_F_4 / temperature, TS_F_4 / temperature, gts, 4)
        #     TS_loss_F = TS_loss_F_1 * 4 + TS_loss_F_2 + TS_loss_F_3 + TS_loss_F_4
        #
        #     T1_loss_F_1 = AT_Loss(S_F_1 / temperature, T1_F_1 / temperature, gts, 4)
        #     T1_loss_F_2 = AT_Loss(S_F_2 / temperature, T1_F_2 / temperature, gts, 4)
        #     T1_loss_F_3 = AT_Loss(S_F_3 / temperature, T1_F_3 / temperature, gts, 4)
        #     T1_loss_F_4 = AT_Loss(S_F_4 / temperature, T1_F_4 / temperature, gts, 4)
        #     T1_loss_F = T1_loss_F_1 * 4 + T1_loss_F_2 + T1_loss_F_3 + T1_loss_F_4
        #
        #     T2_loss_F_1 = AT_Loss(S_F_1 / 4, T2_F_1 / 4, gts, 4)
        #     T2_loss_F_2 = AT_Loss(S_F_2 / 4, T2_F_2 / 4, gts, 4)
        #     T2_loss_F_3 = AT_Loss(S_F_3 / 4, T2_F_3 / 4, gts, 4)
        #     T2_loss_F_4 = AT_Loss(S_F_4 / 4, T2_F_4 / 4, gts, 4)
        #     T2_loss_F = T2_loss_F_1 * 4 + T2_loss_F_2 + T2_loss_F_3 + T2_loss_F_4
        #
        #     T3_loss_F_1 = AT_Loss(S_F_1 / 4, T3_F_1 / 4, gts, 4)
        #     T3_loss_F_2 = AT_Loss(S_F_2 / 4, T3_F_2 / 4, gts, 4)
        #     T3_loss_F_3 = AT_Loss(S_F_3 / 4, T3_F_3 / 4, gts, 4)
        #     T3_loss_F_4 = AT_Loss(S_F_4 / 4, T3_F_4 / 4, gts, 4)
        #     T3_loss_F = T3_loss_F_1 * 4 + T3_loss_F_2 + T3_loss_F_3 + T3_loss_F_4
        #
        #     loss_F = TS_loss_F + T1_loss_F + T2_loss_F + T3_loss_F
        #
        #     loss_gt1 = CE(S_S_1, gts)
        #     loss_gt2 = CE(S_S_2, gts)
        #     loss_gt = loss_gt1 * 4 + loss_gt2
        #     loss_kd_b = 0
        #
        #     TS_loss_kd_1 = KLDLoss1(S_S_1 / temperature, TS_S_1 / temperature, gts, 4) * 4
        #     TS_loss_kd_2 = KLDLoss2(S_S_2 / temperature, TS_S_2 / temperature, gts, 4)
        #     TS_loss_kd_3 = KLDLoss3(S_S_3 / temperature, TS_S_3 / temperature, gts, 4)
        #     TS_loss_kd_4 = KLDLoss4(S_S_4 / temperature, TS_S_4 / temperature, gts, 4)
        #     TS_loss_kd = (TS_loss_kd_1 + TS_loss_kd_2 + TS_loss_kd_3 + TS_loss_kd_4) * (1 - 0.0625) + 0.0625 * loss_gt
        #
        #     T1_loss_kd_1 = KLDLoss1(S_S_1 / temperature, T1_S_1 / temperature, gts, 4) * 4
        #     T1_loss_kd_2 = KLDLoss2(S_S_2 / temperature, T1_S_2 / temperature, gts, 4)
        #     T1_loss_kd_3 = KLDLoss3(S_S_3 / temperature, T1_S_3 / temperature, gts, 4)
        #     T1_loss_kd_4 = KLDLoss4(S_S_4 / temperature, T1_S_4 / temperature, gts, 4)
        #     T1_loss_kd = (T1_loss_kd_1 + T1_loss_kd_2 + T1_loss_kd_3 + T1_loss_kd_4) * (1 - 0.0625) + 0.0625 * loss_gt
        #
        #     T2_loss_kd_1 = KLDLoss1(S_S_1 / 4, T2_S_1 / 4, gts, 4) * 4
        #     T2_loss_kd_2 = KLDLoss2(S_S_2 / 4, T2_S_2 / 4, gts, 4)
        #     T2_loss_kd_3 = KLDLoss3(S_S_3 / 4, T2_S_3 / 4, gts, 4)
        #     T2_loss_kd_4 = KLDLoss4(S_S_4 / 4, T2_S_4 / 4, gts, 4)
        #     T2_loss_kd = (T2_loss_kd_1 + T2_loss_kd_2 + T2_loss_kd_3 + T2_loss_kd_4) * (1 - 0.0625) + 0.0625 * loss_gt
        #
        #     T3_loss_kd_1 = KLDLoss1(S_S_1 / 4, T3_S_1 / 4, gts, 4) * 4
        #     T3_loss_kd_2 = KLDLoss2(S_S_2 / 4, T3_S_2 / 4, gts, 4)
        #     T3_loss_kd_3 = KLDLoss3(S_S_3 / 4, T3_S_3 / 4, gts, 4)
        #     T3_loss_kd_4 = KLDLoss4(S_S_4 / 4, T3_S_4 / 4, gts, 4)
        #     T3_loss_kd = (T3_loss_kd_1 + T3_loss_kd_2 + T3_loss_kd_3 + T3_loss_kd_4) * (1 - 0.0625) + 0.0625 * loss_gt
        #
        #     loss_kd = TS_loss_kd + T1_loss_kd + T2_loss_kd + T3_loss_kd
        #
        # elif 100 < epoch <= 130:
        #     with torch.no_grad():
        #         TS_S_1, TS_S_2, TS_S_3, TS_S_4, TS_F_1, TS_F_2, TS_F_3, TS_F_4, TS_R_4, \
        #             TS_D_4, TS_r_1, TS_r_2, TS_r_3, TS_r_4, TS_d_1, TS_d_2, TS_d_3, TS_d_4 = model_T_S(inputs, depths)
        #
        #         T1_S_1, T1_S_2, T1_S_3, T1_S_4, T1_F_1, T1_F_2, T1_F_3, T1_F_4, T1_R_4, \
        #             T1_D_4, T1_r_1, T1_r_2, T1_r_3, T1_r_4, T1_d_1, T1_d_2, T1_d_3, T1_d_4 = model_T1(inputs, depths)
        #
        #         T2_S_1, T2_S_2, T2_S_3, T2_S_4, T2_F_1, T2_F_2, T2_F_3, T2_F_4, T2_R_4, \
        #             T2_D_4, T2_r_1, T2_r_2, T2_r_3, T2_r_4, T2_d_1, T2_d_2, T2_d_3, T2_d_4 = model_T2(inputs, depths)
        #
        #         T3_S_1, T3_S_2, T3_S_3, T3_S_4, T3_F_1, T3_F_2, T3_F_3, T3_F_4, T3_R_4, \
        #             T3_D_4, T3_r_1, T3_r_2, T3_r_3, T3_r_4, T3_d_1, T3_d_2, T3_d_3, T3_d_4 = model_T3(inputs, depths)
        #
        #         T4_S_1, T4_S_2, T4_S_3, T4_S_4, T4_F_1, T4_F_2, T4_F_3, T4_F_4, T4_R_4, \
        #             T4_D_4, T4_r_1, T4_r_2, T4_r_3, T4_r_4, T4_d_1, T4_d_2, T4_d_3, T4_d_4 = model_T(inputs, depths)
        #
        #     TS_loss_encoder_R_4 = SP(S_R_4 / 4, TS_R_4 / 4).cuda() * 4
        #     TS_loss_encoder_D_4 = SP(S_D_4 / 4, TS_D_4 / 4).cuda() * 4
        #
        #     T1_loss_encoder_R_4 = SP(S_R_4 / 4, T1_R_4 / 4).cuda() * 4
        #     T1_loss_encoder_D_4 = SP(S_D_4 / 4, T1_D_4 / 4).cuda() * 4
        #
        #     T2_loss_encoder_R_4 = SP(S_R_4 / 4, T2_R_4 / 4).cuda() * 4
        #     T2_loss_encoder_D_4 = SP(S_D_4 / 4, T2_D_4 / 4).cuda() * 4
        #
        #     T3_loss_encoder_R_4 = SP(S_R_4 / 4, T3_R_4 / 4).cuda() * 4
        #     T3_loss_encoder_D_4 = SP(S_D_4 / 4, T3_D_4 / 4).cuda() * 4
        #
        #     T4_loss_encoder_R_4 = SP(S_R_4 / 4, T4_R_4 / 4).cuda() * 4
        #     T4_loss_encoder_D_4 = SP(S_D_4 / 4, T4_D_4 / 4).cuda() * 4
        #
        #     TS_loss_encoder = TS_loss_encoder_R_4 + TS_loss_encoder_D_4
        #     T1_loss_encoder = T1_loss_encoder_R_4 + T1_loss_encoder_D_4
        #     T2_loss_encoder = T2_loss_encoder_R_4 + T2_loss_encoder_D_4
        #     T3_loss_encoder = T3_loss_encoder_R_4 + T3_loss_encoder_D_4
        #     T4_loss_encoder = T4_loss_encoder_R_4 + T4_loss_encoder_D_4
        #
        #     loss_encoder = TS_loss_encoder + T1_loss_encoder + T2_loss_encoder + T3_loss_encoder + T4_loss_encoder / 2
        #
        #     TS_loss_F_1 = AT_Loss(S_F_1 / temperature, TS_F_1 / temperature, gts, 4)
        #     TS_loss_F_2 = AT_Loss(S_F_2 / temperature, TS_F_2 / temperature, gts, 4)
        #     TS_loss_F_3 = AT_Loss(S_F_3 / temperature, TS_F_3 / temperature, gts, 4)
        #     TS_loss_F_4 = AT_Loss(S_F_4 / temperature, TS_F_4 / temperature, gts, 4)
        #     TS_loss_F = TS_loss_F_1 * 4 + TS_loss_F_2 + TS_loss_F_3 + TS_loss_F_4
        #
        #     T1_loss_F_1 = AT_Loss(S_F_1 / temperature, T1_F_1 / temperature, gts, 4)
        #     T1_loss_F_2 = AT_Loss(S_F_2 / temperature, T1_F_2 / temperature, gts, 4)
        #     T1_loss_F_3 = AT_Loss(S_F_3 / temperature, T1_F_3 / temperature, gts, 4)
        #     T1_loss_F_4 = AT_Loss(S_F_4 / temperature, T1_F_4 / temperature, gts, 4)
        #     T1_loss_F = T1_loss_F_1 * 4 + T1_loss_F_2 + T1_loss_F_3 + T1_loss_F_4
        #
        #     T2_loss_F_1 = AT_Loss(S_F_1 / temperature, T2_F_1 / temperature, gts, 4)
        #     T2_loss_F_2 = AT_Loss(S_F_2 / temperature, T2_F_2 / temperature, gts, 4)
        #     T2_loss_F_3 = AT_Loss(S_F_3 / temperature, T2_F_3 / temperature, gts, 4)
        #     T2_loss_F_4 = AT_Loss(S_F_4 / temperature, T2_F_4 / temperature, gts, 4)
        #     T2_loss_F = T2_loss_F_1 * 4 + T2_loss_F_2 + T2_loss_F_3 + T2_loss_F_4
        #
        #     T3_loss_F_1 = AT_Loss(S_F_1 / 4, T3_F_1 / 4, gts, 4)
        #     T3_loss_F_2 = AT_Loss(S_F_2 / 4, T3_F_2 / 4, gts, 4)
        #     T3_loss_F_3 = AT_Loss(S_F_3 / 4, T3_F_3 / 4, gts, 4)
        #     T3_loss_F_4 = AT_Loss(S_F_4 / 4, T3_F_4 / 4, gts, 4)
        #     T3_loss_F = T3_loss_F_1 * 4 + T3_loss_F_2 + T3_loss_F_3 + T3_loss_F_4
        #
        #     T4_loss_F_1 = AT_Loss(S_F_1 / 4, T4_F_1 / 4, gts, 4)
        #     T4_loss_F_2 = AT_Loss(S_F_2 / 4, T4_F_2 / 4, gts, 4)
        #     T4_loss_F_3 = AT_Loss(S_F_3 / 4, T4_F_3 / 4, gts, 4)
        #     T4_loss_F_4 = AT_Loss(S_F_4 / 4, T4_F_4 / 4, gts, 4)
        #     T4_loss_F = T4_loss_F_1 * 4 + T4_loss_F_2 + T4_loss_F_3 + T4_loss_F_4
        #
        #     loss_F = TS_loss_F + T1_loss_F + T2_loss_F + T3_loss_F + T4_loss_F
        #
        #     loss_gt1 = CE(S_S_1, gts)
        #     loss_gt2 = CE(S_S_2, gts)
        #     loss_gt = loss_gt1 * 4 + loss_gt2
        #     loss_kd_b = 0
        #
        #     TS_loss_kd_1 = KLDLoss1(S_S_1 / 4, TS_S_1 / 4, gts, 4) * 4
        #     TS_loss_kd_2 = KLDLoss2(S_S_2 / 4, TS_S_2 / 4, gts, 4)
        #     TS_loss_kd_3 = KLDLoss3(S_S_3 / 4, TS_S_3 / 4, gts, 4)
        #     TS_loss_kd_4 = KLDLoss4(S_S_4 / 4, TS_S_4 / 4, gts, 4)
        #     TS_loss_kd = (TS_loss_kd_1 + TS_loss_kd_2 + TS_loss_kd_3 + TS_loss_kd_4) * (1 - 0.0625) + 0.0625 * loss_gt
        #
        #     T1_loss_kd_1 = KLDLoss1(S_S_1 / temperature, T1_S_1 / temperature, gts, 4) * 4
        #     T1_loss_kd_2 = KLDLoss2(S_S_2 / temperature, T1_S_2 / temperature, gts, 4)
        #     T1_loss_kd_3 = KLDLoss3(S_S_3 / temperature, T1_S_3 / temperature, gts, 4)
        #     T1_loss_kd_4 = KLDLoss4(S_S_4 / temperature, T1_S_4 / temperature, gts, 4)
        #     T1_loss_kd = (T1_loss_kd_1 + T1_loss_kd_2 + T1_loss_kd_3 + T1_loss_kd_4) * (1 - 0.0625) + 0.0625 * loss_gt
        #
        #     T2_loss_kd_1 = KLDLoss1(S_S_1 / temperature, T2_S_1 / temperature, gts, 4) * 4
        #     T2_loss_kd_2 = KLDLoss2(S_S_2 / temperature, T2_S_2 / temperature, gts, 4)
        #     T2_loss_kd_3 = KLDLoss3(S_S_3 / temperature, T2_S_3 / temperature, gts, 4)
        #     T2_loss_kd_4 = KLDLoss4(S_S_4 / temperature, T2_S_4 / temperature, gts, 4)
        #     T2_loss_kd = (T2_loss_kd_1 + T2_loss_kd_2 + T2_loss_kd_3 + T2_loss_kd_4) * (1 - 0.0625) + 0.0625 * loss_gt
        #
        #     T3_loss_kd_1 = KLDLoss1(S_S_1 / temperature, T3_S_1 / temperature, gts, 4) * 4
        #     T3_loss_kd_2 = KLDLoss2(S_S_2 / temperature, T3_S_2 / temperature, gts, 4)
        #     T3_loss_kd_3 = KLDLoss3(S_S_3 / temperature, T3_S_3 / temperature, gts, 4)
        #     T3_loss_kd_4 = KLDLoss4(S_S_4 / temperature, T3_S_4 / temperature, gts, 4)
        #     T3_loss_kd = (T3_loss_kd_1 + T3_loss_kd_2 + T3_loss_kd_3 + T3_loss_kd_4) * (1 - 0.0625) + 0.0625 * loss_gt
        #
        #     T4_loss_kd_1 = KLDLoss1(S_S_1 / 4, T4_S_1 / 4, gts, 4) * 4
        #     T4_loss_kd_2 = KLDLoss2(S_S_2 / 4, T4_S_2 / 4, gts, 4)
        #     T4_loss_kd_3 = KLDLoss3(S_S_3 / 4, T4_S_3 / 4, gts, 4)
        #     T4_loss_kd_4 = KLDLoss4(S_S_4 / 4, T4_S_4 / 4, gts, 4)
        #     T4_loss_kd = (T4_loss_kd_1 + T4_loss_kd_2 + T4_loss_kd_3 + T4_loss_kd_4) * (1 - 0.0625) + 0.0625 * loss_gt
        #
        #     loss_kd = TS_loss_kd + T1_loss_kd + T2_loss_kd + T3_loss_kd + T4_loss_kd
        #
        # elif 130 < epoch <= 160:
        #     with torch.no_grad():
        #         TS_S_1, TS_S_2, TS_S_3, TS_S_4, TS_F_1, TS_F_2, TS_F_3, TS_F_4, TS_R_4, \
        #             TS_D_4, TS_r_1, TS_r_2, TS_r_3, TS_r_4, TS_d_1, TS_d_2, TS_d_3, TS_d_4 = model_T_S(inputs, depths)
        #
        #         T1_S_1, T1_S_2, T1_S_3, T1_S_4, T1_F_1, T1_F_2, T1_F_3, T1_F_4, T1_R_4, \
        #             T1_D_4, T1_r_1, T1_r_2, T1_r_3, T1_r_4, T1_d_1, T1_d_2, T1_d_3, T1_d_4 = model_T1(inputs, depths)
        #
        #         T2_S_1, T2_S_2, T2_S_3, T2_S_4, T2_F_1, T2_F_2, T2_F_3, T2_F_4, T2_R_4, \
        #             T2_D_4, T2_r_1, T2_r_2, T2_r_3, T2_r_4, T2_d_1, T2_d_2, T2_d_3, T2_d_4 = model_T2(inputs, depths)
        #
        #         T3_S_1, T3_S_2, T3_S_3, T3_S_4, T3_F_1, T3_F_2, T3_F_3, T3_F_4, T3_R_4, \
        #             T3_D_4, T3_r_1, T3_r_2, T3_r_3, T3_r_4, T3_d_1, T3_d_2, T3_d_3, T3_d_4 = model_T3(inputs, depths)
        #
        #         T4_S_1, T4_S_2, T4_S_3, T4_S_4, T4_F_1, T4_F_2, T4_F_3, T4_F_4, T4_R_4, \
        #             T4_D_4, T4_r_1, T4_r_2, T4_r_3, T4_r_4, T4_d_1, T4_d_2, T4_d_3, T4_d_4 = model_T(inputs, depths)
        #
        #     TS_loss_encoder_R_4 = SP(S_R_4 / 4, TS_R_4 / 4).cuda() * 4
        #     TS_loss_encoder_D_4 = SP(S_D_4 / 4, TS_D_4 / 4).cuda() * 4
        #
        #     T1_loss_encoder_R_4 = SP(S_R_4 / 4, T1_R_4 / 4).cuda() * 4
        #     T1_loss_encoder_D_4 = SP(S_D_4 / 4, T1_D_4 / 4).cuda() * 4
        #
        #     T2_loss_encoder_R_4 = SP(S_R_4 / 4, T2_R_4 / 4).cuda() * 4
        #     T2_loss_encoder_D_4 = SP(S_D_4 / 4, T2_D_4 / 4).cuda() * 4
        #
        #     T3_loss_encoder_R_4 = SP(S_R_4 / 4, T3_R_4 / 4).cuda() * 4
        #     T3_loss_encoder_D_4 = SP(S_D_4 / 4, T3_D_4 / 4).cuda() * 4
        #
        #     T4_loss_encoder_R_4 = SP(S_R_4 / 4, T4_R_4 / 4).cuda() * 4
        #     T4_loss_encoder_D_4 = SP(S_D_4 / 4, T4_D_4 / 4).cuda() * 4
        #
        #     TS_loss_encoder = TS_loss_encoder_R_4 + TS_loss_encoder_D_4
        #     T1_loss_encoder = T1_loss_encoder_R_4 + T1_loss_encoder_D_4
        #     T2_loss_encoder = T2_loss_encoder_R_4 + T2_loss_encoder_D_4
        #     T3_loss_encoder = T3_loss_encoder_R_4 + T3_loss_encoder_D_4
        #     T4_loss_encoder = T4_loss_encoder_R_4 + T4_loss_encoder_D_4
        #
        #     loss_encoder = TS_loss_encoder + T1_loss_encoder + T2_loss_encoder + T3_loss_encoder + T4_loss_encoder
        #
        #     TS_loss_F_1 = AT_Loss(S_F_1 / temperature, TS_F_1 / temperature, gts, 4)
        #     TS_loss_F_2 = AT_Loss(S_F_2 / temperature, TS_F_2 / temperature, gts, 4)
        #     TS_loss_F_3 = AT_Loss(S_F_3 / temperature, TS_F_3 / temperature, gts, 4)
        #     TS_loss_F_4 = AT_Loss(S_F_4 / temperature, TS_F_4 / temperature, gts, 4)
        #     TS_loss_F = TS_loss_F_1 * 4 + TS_loss_F_2 + TS_loss_F_3 + TS_loss_F_4
        #
        #     T1_loss_F_1 = AT_Loss(S_F_1 / temperature, T1_F_1 / temperature, gts, 4)
        #     T1_loss_F_2 = AT_Loss(S_F_2 / temperature, T1_F_2 / temperature, gts, 4)
        #     T1_loss_F_3 = AT_Loss(S_F_3 / temperature, T1_F_3 / temperature, gts, 4)
        #     T1_loss_F_4 = AT_Loss(S_F_4 / temperature, T1_F_4 / temperature, gts, 4)
        #     T1_loss_F = T1_loss_F_1 * 4 + T1_loss_F_2 + T1_loss_F_3 + T1_loss_F_4
        #
        #     T2_loss_F_1 = AT_Loss(S_F_1 / temperature, T2_F_1 / temperature, gts, 4)
        #     T2_loss_F_2 = AT_Loss(S_F_2 / temperature, T2_F_2 / temperature, gts, 4)
        #     T2_loss_F_3 = AT_Loss(S_F_3 / temperature, T2_F_3 / temperature, gts, 4)
        #     T2_loss_F_4 = AT_Loss(S_F_4 / temperature, T2_F_4 / temperature, gts, 4)
        #     T2_loss_F = T2_loss_F_1 * 4 + T2_loss_F_2 + T2_loss_F_3 + T2_loss_F_4
        #
        #     T3_loss_F_1 = AT_Loss(S_F_1 / temperature, T3_F_1 / temperature, gts, 4)
        #     T3_loss_F_2 = AT_Loss(S_F_2 / temperature, T3_F_2 / temperature, gts, 4)
        #     T3_loss_F_3 = AT_Loss(S_F_3 / temperature, T3_F_3 / temperature, gts, 4)
        #     T3_loss_F_4 = AT_Loss(S_F_4 / temperature, T3_F_4 / temperature, gts, 4)
        #     T3_loss_F = T3_loss_F_1 * 4 + T3_loss_F_2 + T3_loss_F_3 + T3_loss_F_4
        #
        #     T4_loss_F_1 = AT_Loss(S_F_1 / temperature, T4_F_1 / temperature, gts, 4)
        #     T4_loss_F_2 = AT_Loss(S_F_2 / temperature, T4_F_2 / temperature, gts, 4)
        #     T4_loss_F_3 = AT_Loss(S_F_3 / temperature, T4_F_3 / temperature, gts, 4)
        #     T4_loss_F_4 = AT_Loss(S_F_4 / temperature, T4_F_4 / temperature, gts, 4)
        #     T4_loss_F = T4_loss_F_1 * 4 + T4_loss_F_2 + T4_loss_F_3 + T4_loss_F_4
        #
        #     loss_F = TS_loss_F + T1_loss_F + T2_loss_F + T3_loss_F + T4_loss_F
        #
        #     loss_gt1 = CE(S_S_1, gts)
        #     loss_gt2 = CE(S_S_2, gts)
        #     loss_gt = loss_gt1 * 4 + loss_gt2
        #     loss_kd_b = 0
        #
        #     TS_loss_kd_1 = KLDLoss1(S_S_1 / temperature, TS_S_1 / temperature, gts, 4) * 4
        #     TS_loss_kd_2 = KLDLoss2(S_S_2 / temperature, TS_S_2 / temperature, gts, 4)
        #     TS_loss_kd_3 = KLDLoss3(S_S_3 / temperature, TS_S_3 / temperature, gts, 4)
        #     TS_loss_kd_4 = KLDLoss4(S_S_4 / temperature, TS_S_4 / temperature, gts, 4)
        #     TS_loss_kd = (TS_loss_kd_1 + TS_loss_kd_2 + TS_loss_kd_3 + TS_loss_kd_4) * (1 - 0.0625) + 0.0625 * loss_gt
        #
        #     T1_loss_kd_1 = KLDLoss1(S_S_1 / temperature, T1_S_1 / temperature, gts, 4) * 4
        #     T1_loss_kd_2 = KLDLoss2(S_S_2 / temperature, T1_S_2 / temperature, gts, 4)
        #     T1_loss_kd_3 = KLDLoss3(S_S_3 / temperature, T1_S_3 / temperature, gts, 4)
        #     T1_loss_kd_4 = KLDLoss4(S_S_4 / temperature, T1_S_4 / temperature, gts, 4)
        #     T1_loss_kd = (T1_loss_kd_1 + T1_loss_kd_2 + T1_loss_kd_3 + T1_loss_kd_4) * (1 - 0.0625) + 0.0625 * loss_gt
        #
        #     T2_loss_kd_1 = KLDLoss1(S_S_1 / temperature, T2_S_1 / temperature, gts, 4) * 4
        #     T2_loss_kd_2 = KLDLoss2(S_S_2 / temperature, T2_S_2 / temperature, gts, 4)
        #     T2_loss_kd_3 = KLDLoss3(S_S_3 / temperature, T2_S_3 / temperature, gts, 4)
        #     T2_loss_kd_4 = KLDLoss4(S_S_4 / temperature, T2_S_4 / temperature, gts, 4)
        #     T2_loss_kd = (T2_loss_kd_1 + T2_loss_kd_2 + T2_loss_kd_3 + T2_loss_kd_4) * (1 - 0.0625) + 0.0625 * loss_gt
        #
        #     T3_loss_kd_1 = KLDLoss1(S_S_1 / temperature, T3_S_1 / temperature, gts, 4) * 4
        #     T3_loss_kd_2 = KLDLoss2(S_S_2 / temperature, T3_S_2 / temperature, gts, 4)
        #     T3_loss_kd_3 = KLDLoss3(S_S_3 / temperature, T3_S_3 / temperature, gts, 4)
        #     T3_loss_kd_4 = KLDLoss4(S_S_4 / temperature, T3_S_4 / temperature, gts, 4)
        #     T3_loss_kd = (T3_loss_kd_1 + T3_loss_kd_2 + T3_loss_kd_3 + T3_loss_kd_4) * (1 - 0.0625) + 0.0625 * loss_gt
        #
        #     T4_loss_kd_1 = KLDLoss1(S_S_1 / temperature, T4_S_1 / temperature, gts, 4) * 4
        #     T4_loss_kd_2 = KLDLoss2(S_S_2 / temperature, T4_S_2 / temperature, gts, 4)
        #     T4_loss_kd_3 = KLDLoss3(S_S_3 / temperature, T4_S_3 / temperature, gts, 4)
        #     T4_loss_kd_4 = KLDLoss4(S_S_4 / temperature, T4_S_4 / temperature, gts, 4)
        #     T4_loss_kd = (T4_loss_kd_1 + T4_loss_kd_2 + T4_loss_kd_3 + T4_loss_kd_4) * (1 - 0.0625) + 0.0625 * loss_gt
        #
        #     loss_kd = TS_loss_kd + T1_loss_kd + T2_loss_kd + T3_loss_kd + T4_loss_kd
        #
        # else:
        #     T4_S_1, T4_S_2, T4_S_3, T4_S_4, T4_F_1, T4_F_2, T4_F_3, T4_F_4, T4_R_4, \
        #         T4_D_4, T4_r_1, T4_r_2, T4_r_3, T4_r_4, T4_d_1, T4_d_2, T4_d_3, T4_d_4 = model_T(inputs, depths)
        #
        #
        #     T4_loss_encoder_R_4 = SP(S_R_4 / 4, T4_R_4 / 4).cuda() * 4
        #     T4_loss_encoder_D_4 = SP(S_D_4 / 4, T4_D_4 / 4).cuda() * 4
        #     T4_loss_encoder = T4_loss_encoder_R_4 + T4_loss_encoder_D_4
        #     loss_encoder = T4_loss_encoder
        #
        #     T4_loss_F_1 = AT_Loss(S_F_1 / temperature, T4_F_1 / temperature, gts, 4)
        #     T4_loss_F_2 = AT_Loss(S_F_2 / temperature, T4_F_2 / temperature, gts, 4)
        #     T4_loss_F_3 = AT_Loss(S_F_3 / temperature, T4_F_3 / temperature, gts, 4)
        #     T4_loss_F_4 = AT_Loss(S_F_4 / temperature, T4_F_4 / temperature, gts, 4)
        #     T4_loss_F = T4_loss_F_1 * 4 + T4_loss_F_2 + T4_loss_F_3 + T4_loss_F_4
        #
        #     loss_F = T4_loss_F
        #
        #     loss_gt1 = CE(S_S_1, gts)
        #     loss_gt2 = CE(S_S_2, gts)
        #     loss_gt = loss_gt1 * 4 + loss_gt2
        #     loss_kd_b = 0
        #
        #     T4_loss_kd_1 = KLDLoss1(S_S_1 / temperature, T4_S_1 / temperature, gts, 4) * 4
        #     T4_loss_kd_2 = KLDLoss2(S_S_2 / temperature, T4_S_2 / temperature, gts, 4)
        #     T4_loss_kd_3 = KLDLoss3(S_S_3 / temperature, T4_S_3 / temperature, gts, 4)
        #     T4_loss_kd_4 = KLDLoss4(S_S_4 / temperature, T4_S_4 / temperature, gts, 4)
        #     T4_loss_kd = (T4_loss_kd_1 + T4_loss_kd_2 + T4_loss_kd_3 + T4_loss_kd_4) * (1 - 0.0625) + 0.0625 * loss_gt
        #
        #     loss_kd = T4_loss_kd
        #
        #
        # loss_sal = loss_kd * 1 + loss_gt * 1 + loss_F * 1 + loss_encoder * 1 + loss_kd_b
        # loss_sal = loss_sal * 1

        with torch.no_grad():
            # false_values = model_T(inputs, depths)
            T_S_1, T_S_2, T_S_3, T_S_4, T_F_1, T_F_2, T_F_3, T_F_4, T_R_4, \
            T_D_4, T_r_1, T_r_2, T_r_3, T_r_4, T_d_1, T_d_2, T_d_3, T_d_4 = model_T(inputs, depths)

            # TS_S_1, TS_S_2, TS_S_3, TS_S_4, TS_F_1, TS_F_2, TS_F_3, TS_F_4, TS_R_4, \
            #     TS_D_4, TS_r_1, TS_r_2, TS_r_3, TS_r_4, TS_d_1, TS_d_2, TS_d_3, TS_d_4 = model_T_S(inputs, depths)

        # false_labels = model_T(inputs, depths)
        # return_values = model(inputs, depths)
        S_S_1, S_S_2, S_S_3, S_S_4, S_F_1, S_F_2, S_F_3, S_F_4, S_R_4, \
        S_D_4, S_r_1, S_r_2, S_r_3, S_r_4, S_d_1, S_d_2, S_d_3, S_d_4 = model(inputs, depths)

        s_s_1 = F.sigmoid(S_S_1)
        s_s_2 = F.sigmoid(S_S_2)
        s_s_3 = F.sigmoid(S_S_3)
        s_s_4 = F.sigmoid(S_S_4)

        t_s_1 = F.sigmoid(T_S_1)
        t_s_2 = F.sigmoid(T_S_2)
        t_s_3 = F.sigmoid(T_S_3)
        t_s_4 = F.sigmoid(T_S_4)

        # ts_s_1 = F.sigmoid(TS_S_1)
        # ts_s_2 = F.sigmoid(TS_S_2)
        # ts_s_3 = F.sigmoid(TS_S_3)
        # ts_s_4 = F.sigmoid(TS_S_4)

        s_s_1 = 1 - s_s_1
        s_s_2 = 1 - s_s_2
        s_s_3 = 1 - s_s_3
        s_s_4 = 1 - s_s_4

        t_s_1 = 1 - t_s_1
        t_s_2 = 1 - t_s_2
        t_s_3 = 1 - t_s_3
        t_s_4 = 1 - t_s_4

        # ts_s_1 = 1 - ts_s_1
        # ts_s_2 = 1 - ts_s_2
        # ts_s_3 = 1 - ts_s_3
        # ts_s_4 = 1 - ts_s_4

        loss = 0

        if epoch <= 30:

            loss5 = CE(S_S_1, gts)
            loss6 = CE(S_S_2, gts)
            loss7 = CE(S_S_3, gts)
            loss8 = CE(S_S_4, gts)
            loss_gt = loss5 * 4 + loss6 + loss7 + loss8

            los1_8 = KLDLoss1(S_S_1 / 4, T_S_1 / 4, gts, 5)
            los2_8 = KLDLoss2(S_S_2 / 4, T_S_2 / 4, gts, 5)
            los3_8 = KLDLoss3(S_S_3 / 4, T_S_3 / 4, gts, 5)
            los4_8 = KLDLoss4(S_S_4 / 4, T_S_4 / 4, gts, 5)
            loss_final = los1_8 * 4 + los2_8 + los3_8 + los4_8
            loss_kd = loss_final * (1 - 0.0625) + 0.0625 * loss_gt

            loss_kd = loss_kd


            loss_kd_b1 = KLDLoss1(s_s_1 / temperature, t_s_1 / temperature, gts, 4)
            loss_kd_b2 = KLDLoss1(s_s_2 / temperature, t_s_2 / temperature, gts, 4)
            loss_kd_b3 = KLDLoss1(s_s_3 / temperature, t_s_3 / temperature, gts, 4)
            loss_kd_b4 = KLDLoss1(s_s_4 / temperature, t_s_4 / temperature, gts, 4)
            loss_kd_b = loss_kd_b1 * 4 + loss_kd_b2 + loss_kd_b3 + loss_kd_b4

            loss_kd_b = loss_kd_b

            loss_encoder_r4 = SP(S_R_4 / 4, T_R_4 / 4).cuda() * 4
            loss_encoder_d4 = SP(S_D_4 / 4, T_D_4 / 4).cuda() * 4
            loss_decoder_r1 = SP(S_r_1 / temperature, T_r_1 / temperature).cuda()
            loss_decoder_r2 = SP(S_r_2 / temperature, T_r_2 / temperature).cuda()
            loss_decoder_r3 = SP(S_r_3 / temperature, T_r_3 / temperature).cuda()
            loss_decoder_r4 = SP(S_r_4 / temperature, T_r_4 / temperature).cuda()
            loss_decoder_d1 = SP(S_d_1 / temperature, T_d_1 / temperature).cuda()
            loss_decoder_d2 = SP(S_d_2 / temperature, T_d_2 / temperature).cuda()
            loss_decoder_d3 = SP(S_d_3 / temperature, T_d_3 / temperature).cuda()
            loss_decoder_d4 = SP(S_d_4 / temperature, T_d_4 / temperature).cuda()

            loss_encoder = loss_encoder_r4 + loss_encoder_d4 + loss_decoder_r1 + loss_decoder_r2 + loss_decoder_r3 + \
                             loss_decoder_r4 + loss_decoder_d1 + loss_decoder_d2 + loss_decoder_d3 + loss_decoder_d4

            loss_encoder = loss_encoder

            loss_F_1 = AT_Loss(S_F_1 / temperature, T_F_1 / temperature, gts, 4)
            loss_F_2 = AT_Loss(S_F_2 / temperature, T_F_2 / temperature, gts, 4)
            loss_F_3 = AT_Loss(S_F_3 / temperature, T_F_3 / temperature, gts, 4)
            loss_F_4 = AT_Loss(S_F_4 / temperature, T_F_4 / temperature, gts, 4)
            loss_midel = loss_F_1 * 4 + loss_F_2 + loss_F_3 + loss_F_4

            loss_midel = loss_midel

        elif 30 < epoch <=60:
            loss5 = CE(S_S_1, gts)
            loss6 = CE(S_S_2, gts)
            loss7 = 0
            loss8 = 0
            loss_gt = loss5 * 4 + loss6 + loss7 + loss8

            los1_8 = KLDLoss1(S_S_1 / 1, T_S_1 / 1, gts, 5)
            los2_8 = KLDLoss2(S_S_2 / 1, T_S_2 / 1, gts, 5)
            los3_8 = KLDLoss3(S_S_3 / 1, T_S_3 / 1, gts, 5)
            los4_8 = KLDLoss4(S_S_4 / 1, T_S_4 / 1, gts, 5)
            loss_final = los1_8 * 4 + los2_8 + los3_8 + los4_8
            loss_kd = loss_final * (1 - 0.0625) + 0.0625 * loss_gt

            loss_kd = loss_kd

            loss_kd_b1 = KLDLoss1(s_s_1 / 4, t_s_1 / 4, gts, 4)
            loss_kd_b2 = KLDLoss1(s_s_2 / 4, t_s_2 / 4, gts, 4)
            loss_kd_b3 = KLDLoss1(s_s_3 / 4, t_s_3 / 4, gts, 4)
            loss_kd_b4 = KLDLoss1(s_s_4 / 4, t_s_4 / 4, gts, 4)
            loss_kd_b = loss_kd_b1 * 4 + loss_kd_b2 + loss_kd_b3 + loss_kd_b4


            loss_kd_b = loss_kd_b

            loss_encoder_r4 = SP(S_R_4 / 4, T_R_4 / 4).cuda() * 4
            loss_encoder_d4 = SP(S_D_4 / 4, T_D_4 / 4).cuda() * 4

            loss_encoder = loss_encoder_r4 + loss_encoder_d4
            loss_encoder = loss_encoder

            loss_F_1 = AT_Loss(S_F_1 / 1, T_F_1 / 1, gts, 4)
            loss_F_2 = AT_Loss(S_F_2 / 1, T_F_2 / 1, gts, 4)
            loss_F_3 = AT_Loss(S_F_3 / 1, T_F_3 / 1, gts, 4)
            loss_F_4 = AT_Loss(S_F_4 / 1, T_F_4 / 1, gts, 4)
            loss_midel = loss_F_1 * 4 + loss_F_2 + loss_F_3 + loss_F_4
            loss_midel = loss_midel

        elif 60 < epoch <=90:
            loss5 = CE(S_S_1, gts)
            loss6 = CE(S_S_2, gts)
            loss7 = 0
            loss8 = 0
            loss_gt = loss5 * 4 + loss6 + loss7 + loss8

            los1_8 = KLDLoss1(S_S_1 / temperature, T_S_1 / temperature, gts, 5)
            los2_8 = KLDLoss2(S_S_2 / temperature, T_S_2 / temperature, gts, 5)
            los3_8 = KLDLoss3(S_S_3 / temperature, T_S_3 / temperature, gts, 5)
            los4_8 = KLDLoss4(S_S_4 / temperature, T_S_4 / temperature, gts, 5)
            loss_final = los1_8 * 4 + los2_8 + los3_8 + los4_8
            loss_kd = loss_final * (1 - 0.0625) + 0.0625 * loss_gt

            loss_kd = loss_kd

            loss_kd_b = 0

            loss_encoder_r4 = SP(S_R_4 / 4, T_R_4 / 4).cuda() * 4
            loss_encoder_d4 = SP(S_D_4 / 4, T_D_4 / 4).cuda() * 4
            loss_encoder = loss_encoder_r4 + loss_encoder_d4
            loss_encoder = loss_encoder

            loss_F_1 = AT_Loss(S_F_1 / temperature, T_F_1 / temperature, gts, 4)
            loss_F_2 = AT_Loss(S_F_2 / temperature, T_F_2 / temperature, gts, 4)
            loss_F_3 = AT_Loss(S_F_3 / temperature, T_F_3 / temperature, gts, 4)
            loss_F_4 = AT_Loss(S_F_4 / temperature, T_F_4 / temperature, gts, 4)
            loss_midel = loss_F_1 * 4 + loss_F_2 + loss_F_3 + loss_F_4

            loss_midel = loss_midel


        else:
            loss5 = CE(S_S_1, gts)
            loss6 = CE(S_S_2, gts)
            loss7 = 0
            loss8 = 0
            loss_gt = loss5 * 4 + loss6 + loss7 + loss8

            los1_8 = KLDLoss1(S_S_1 / temperature, T_S_1 / temperature, gts, 5)
            los2_8 = KLDLoss2(S_S_2 / temperature, T_S_2 / temperature, gts, 5)
            los3_8 = KLDLoss3(S_S_3 / temperature, T_S_3 / temperature, gts, 5)
            los4_8 = KLDLoss4(S_S_4 / temperature, T_S_4 / temperature, gts, 5)
            loss_final = los1_8 * 4 + los2_8 + los3_8 + los4_8
            loss_kd = loss_final * (1 - 0.0625) + 0.0625 * loss_gt
            loss_kd = loss_kd

            loss_kd_b = 0

            loss_encoder_r4 = SP(S_R_4 / 4, T_R_4 / 4).cuda() * 4
            loss_encoder_d4 = SP(S_D_4 / 4, T_D_4 / 4).cuda() * 4
            loss_encoder = loss_encoder_r4 + loss_encoder_d4
            loss_encoder = loss_encoder

            loss_F_1 = AT_Loss(S_F_1 / temperature, T_F_1 / temperature, gts, 4)
            loss_F_2 = AT_Loss(S_F_2 / temperature, T_F_2 / temperature, gts, 4)
            loss_F_3 = AT_Loss(S_F_3 / temperature, T_F_3 / temperature, gts, 4)
            loss_F_4 = AT_Loss(S_F_4 / temperature, T_F_4 / temperature, gts, 4)
            loss_midel = loss_F_1 * 4 + loss_F_2 + loss_F_3 + loss_F_4

            loss_midel = loss_midel

        loss_sal = loss_kd * 1 + loss_gt * 1 + 1 * loss_midel + loss_encoder * 1 + loss_kd_b


        loss_sal = loss_sal * 1

        loss += loss_sal
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Logger
        if batch_idx % 20 == 0:
            # NOTE: Top2Down; [0] is the grobal slamap and [5] is the final output
            info_progress = 'Epoch[{0}/{1}] Iter[{2}/{3}]'.format(epoch, args.epochs, batch_idx, len(train_loader))
            info_loss = 'Train Loss: loss_sal: {:.3f}'.format(loss_sal)
            logger.info(''.join((info_progress, info_loss)))
    # 对学习率进行更新
    scheduler.step()
    info_loss = '@==Final== Epoch[{0}/{1}]  Train Loss: {loss.avg:.3f}  '.format(epoch, args.epochs, loss=loss_log)
    if config.lambdas_sal_last['triplet']:
        info_loss += 'Triplet Loss: {loss.avg:.3f}  '.format(loss=loss_log_triplet)
    logger.info(info_loss)

    return loss_log.avg


def validate(model, test_loaders, testsets, temperature):
    model.eval()

    testsets = testsets.split('+')
    measures = []
    for testset in testsets[:1]:
        print('Validating {}...'.format(testset))
        test_loader = test_loaders[testset]

        saved_root = os.path.join(args.val_dir, testset)

        for batch in test_loader:

            inputs = batch[0].to(device).squeeze(0)
            # print(inputs.shape)
            gts = batch[1].to(device).squeeze(0)
            depths = batch[2].to(device).squeeze(0)
            edges = batch[3].to(device).squeeze(0)
            subpaths = batch[4]
            ori_sizes = batch[5]
            with torch.no_grad():
                scaled_preds = model(inputs, depths)[0]
                # scaled_preds = model(inputs, temperature)[0]
                # print("hhhh",scaled_preds.size())
            os.makedirs(os.path.join(saved_root, subpaths[0][0].split('/')[0]), exist_ok=True)

            num = len(scaled_preds)
            for inum in range(num):
                subpath = subpaths[inum][0]
                ori_size = (ori_sizes[inum][0].item(), ori_sizes[inum][1].item())
                if config.db_output_refiner or (not config.refine and config.db_output_decoder):
                    res = nn.functional.interpolate(scaled_preds[inum].unsqueeze(0), size=ori_size, mode='bilinear',
                                                    align_corners=True)
                else:
                    res = nn.functional.interpolate(scaled_preds[inum].unsqueeze(0), size=ori_size, mode='bilinear',
                                                    align_corners=True).sigmoid()
                save_tensor_img(res, os.path.join(saved_root, subpath))

        eval_loader = EvalDataset(
            saved_root,  # preds
            os.path.join('/home/maps/Downloads/Alchemist/COA_SOD/Data/gts', testset)  # GT
        )
        evaler = Eval_thread(eval_loader, cuda=True)
        # Use S_measure for validation
        s_measure = evaler.Eval_Smeasure()
        if s_measure > config.val_measures['Smeasure']['val_38'] and 0:  # CoCA
            print("llll")
            # TODO: evluate others measures if s_measure is very high.
            e_max = evaler.Eval_Emeasure().max().item()
            f_max = evaler.Eval_fmeasure().max().item()
            print('Emax: {:4.f}, Fmax: {:4.f}'.format(e_max, f_max))
        measures.append(s_measure)
    model.train()
    return measures


if __name__ == '__main__':
    main()
