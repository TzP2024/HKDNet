import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from COA_RGBD_SOD.al.Loss.losses import *


class PoolingAttention(nn.Module):
    def __init__(self, dim, pool_ratios=[1, 2, 3, 4], num_heads=8):
        super().__init__()
        self.convs = nn.ModuleList((nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim) for temp in pool_ratios))
        self.pool_ratios = pool_ratios
        self.pools = nn.ModuleList()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.q = nn.Sequential(nn.Linear(dim, dim))
        self.kv = nn.Sequential(nn.Linear(dim, dim * 2))
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

        # self.norm = nn.LayerNorm(dim)

    def forward(self, rd):
        B, C, H, W = rd.shape
        x = rd.flatten(2).transpose(1, 2)
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        pools = []
        x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
        for (pool_ratio, l) in zip(self.pool_ratios, self.convs):
            pool = F.adaptive_max_pool2d(x_, (round(H / pool_ratio), round(W / pool_ratio)))
            pool = pool + l(pool)
            pools.append(pool.view(B, C, -1))

        pools = torch.cat(pools, dim=2)
        pools = self.norm(pools.permute(0, 2, 1))
        kv = self.kv(pools).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v)
        x = x.transpose(1, 2).contiguous().reshape(B, N, C)
        x = self.proj(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        return x


class MPSDLoss(nn.Module):
    def __init__(self, student_channels, teacher_channels, pool_ratios=[1, 2, 3, 4], num_heads=8):
        super().__init__()
        self.mse = nn.MSELoss(reduce='mean')
        self.KLD = nn.KLDivLoss(reduction='sum')
        self.teacher_channels = teacher_channels
        self.student_channels = student_channels
        self.conv = nn.Sequential(nn.Conv2d(student_channels, teacher_channels, 1, 1, 0), nn.BatchNorm2d(teacher_channels))
        self.conv_a1 = nn.Sequential(nn.Conv2d(student_channels, teacher_channels, 1, 1, 0), nn.BatchNorm2d(teacher_channels))
        self.conv_a2 = nn.Sequential(nn.Conv2d(student_channels, teacher_channels, 1, 1, 0), nn.BatchNorm2d(teacher_channels))
        self.conv_a3 = nn.Sequential(nn.Conv2d(student_channels, teacher_channels, 1, 1, 0), nn.BatchNorm2d(teacher_channels))
        self.PA1 = PoolingAttention(teacher_channels, pool_ratios, num_heads)
        self.PA2 = PoolingAttention(teacher_channels, pool_ratios, num_heads)
        self.PA3 = PoolingAttention(teacher_channels, pool_ratios, num_heads)


    def forward(self, x_student, x_teacher, x_student_a, x_teacher_a):
        S_B, S_C, S_W, S_H = x_student.shape
        T_B, T_C, T_W, T_H = x_teacher.shape


        x_student1 = x_student_a[0]
        x_student2 = x_student_a[1]
        x_student3 = x_student_a[2]
        loss1 = self.mse(x_student.mean(dim=1), x_teacher.mean(dim=1)) + \
                self.mse(x_student1.mean(dim=1), x_teacher_a[0].mean(dim=1)) + \
                self.mse(x_student2.mean(dim=1), x_teacher_a[1].mean(dim=1)) + \
                self.mse(x_student3.mean(dim=1), x_teacher_a[2].mean(dim=1))

        if S_C != T_C:
            x_student = self.conv(x_student)
            x_student1 = self.conv_a1(x_student1)
            x_student2 = self.conv_a2(x_student2)
            x_student3 = self.conv_a3(x_student3)


        x_student1 = self.PA1(x_student1)
        x_student2 = self.PA2(x_student2)
        x_student3 = self.PA3(x_student3)


        x_teacher1 = x_teacher_a[0]
        x_teacher2 = x_teacher_a[1]
        x_teacher3 = x_teacher_a[2]

        loss2 = self.mse(x_student1.mean(dim=1), x_teacher1.mean(dim=1)) + \
                self.mse(x_student2.mean(dim=1), x_teacher2.mean(dim=1)) + \
                self.mse(x_student3.mean(dim=1), x_teacher3.mean(dim=1))

        x_student = F.log_softmax(x_student, dim=-1)
        x_teacher = F.softmax(x_teacher, dim=-1)

        loss3 = self.KLD(x_student.mean(dim=1), x_teacher.mean(dim=1))/(x_student.numel()/x_student.shape[1])
        loss = loss2 + loss3 + loss1
        return loss





def BDCovpool(x, t):
    batchSize, dim, h, w = x.data.shape
    M = h * w
    x = x.reshape(batchSize, dim, M)

    I = torch.eye(dim, dim, device=x.device).view(1, dim, dim).repeat(batchSize, 1, 1).type(x.dtype)
    I_M = torch.ones(batchSize, dim, dim, device=x.device).type(x.dtype)
    x_pow2 = x.bmm(x.transpose(1, 2))
    dcov = I_M.bmm(x_pow2  * I) + (x_pow2 * I).bmm(I_M) - 2 * x_pow2

    dcov = torch.clamp(dcov, min=0.0)
    dcov = torch.exp(t) * dcov
    dcov = torch.sqrt(dcov + 1e-5)
    t = dcov - 1. / dim * dcov.bmm(I_M) - 1. / dim * I_M.bmm(dcov) + 1. / (dim * dim) * I_M.bmm(dcov).bmm(I_M)
    # print("t shape", t.shape)
    return t



class CDLoss(nn.Module):
    def __init__(self, student_channels, teacher_channels):
        super().__init__()
        self.mse = nn.MSELoss(reduce='mean')
        self.KLD = nn.KLDivLoss(reduction='sum')
        self.teacher_channels = teacher_channels
        self.student_channels = student_channels
        self.conv1 = nn.Sequential(nn.Conv2d(student_channels[0], teacher_channels[0], 1, 1, 0), nn.BatchNorm2d(teacher_channels[0]))
        self.conv2 = nn.Sequential(nn.Conv2d(student_channels[1], teacher_channels[1], 1, 1, 0), nn.BatchNorm2d(teacher_channels[1]))
        self.conv3 = nn.Sequential(nn.Conv2d(student_channels[2], teacher_channels[2], 1, 1, 0), nn.BatchNorm2d(teacher_channels[2]))
        self.temperature = nn.Parameter(torch.log((1. / (teacher_channels[2] * teacher_channels[2])) * torch.ones(1, 1)), requires_grad=True)


    def forward(self, student_fusion1, student_fusion2, student_fusion3, teacher_fusion1, teacher_fusion2, teacher_fusion3):

        loss1 = self.mse(student_fusion1.mean(dim=1), teacher_fusion1.mean(dim=1)) + \
                self.mse(student_fusion2.mean(dim=1), teacher_fusion2.mean(dim=1)) + \
                self.mse(student_fusion3.mean(dim=1), teacher_fusion3.mean(dim=1))

        student_fusion1 = self.conv1(student_fusion1)
        student_fusion2 = self.conv2(student_fusion2)
        student_fusion3 = self.conv3(student_fusion3)

        student_fusion1 = BDCovpool(student_fusion1, self.temperature)
        student_fusion2 = BDCovpool(student_fusion2, self.temperature)
        student_fusion3 = BDCovpool(student_fusion3, self.temperature)

        teacher_fusion1 = BDCovpool(teacher_fusion1, self.temperature)
        teacher_fusion2 = BDCovpool(teacher_fusion2, self.temperature)
        teacher_fusion3 = BDCovpool(teacher_fusion3, self.temperature)

        student_fusion1 = F.log_softmax(student_fusion1, dim=-1)
        teacher_fusion1 = F.softmax(teacher_fusion1, dim=-1)
        student_fusion2 = F.log_softmax(student_fusion2, dim=-1)
        teacher_fusion2 = F.softmax(teacher_fusion2, dim=-1)
        student_fusion3 = F.log_softmax(student_fusion3, dim=-1)
        teacher_fusion3 = F.softmax(teacher_fusion3, dim=-1)

        loss2 = self.KLD(student_fusion1.mean(dim=1), teacher_fusion1.mean(dim=1))/(student_fusion1.numel()/student_fusion1.shape[1]) + \
                self.KLD(student_fusion2.mean(dim=1), teacher_fusion2.mean(dim=1))/(student_fusion2.numel()/student_fusion2.shape[1]) + \
                self.KLD(student_fusion3.mean(dim=1), teacher_fusion3.mean(dim=1))/(student_fusion3.numel()/student_fusion3.shape[1])
        loss = loss1 + loss2
        return loss