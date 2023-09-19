import torch
import numpy as np
import torch.nn.functional as F
import torchio as tio
import torch.optim as optim
import os
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from monai.networks.blocks import Warp

import model
import data
import losses
import utils

task_name = 'exp1'
mode = '4d' # 3d for pair-wise, 4d for group wise
dt = 0.5 # training: 0.2 for pair, 0.5 for group; testing: 0.1 for pair, 0.25 for group
backward=True
scaling=True
global_context=True
log_interval=10
vis_interval=100
checkpoint_interval=2000
image_size = (192, 160, 192)


checkpoint_path = os.path.join('checkpoints', task_name)
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

log_path = os.path.join('log', task_name)
if not os.path.exists(log_path):
    os.makedirs(log_path)
log_writer = SummaryWriter(log_path)


if mode == '4d':
    order1 = [0, 1, 2, 3, 4, 5]
    order2 = [0, 9, 8, 7, 6, 5]

augmentation = tio.Compose([
    tio.RandomBlur(p=0.1),
    tio.RandomGamma(p=0.25),
])

train_dataset = data.ZipDataset(['4d-lung-train', 'spare-train'],
                        mode,
                        augmentation=augmentation,
                        pair=(mode=='3d'), size=image_size)

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)


stn = Warp().cuda()
ncc = losses.NCC().loss
smooth_loss = losses.Grad(penalty='l2').loss

net = model.ODENet(mode=mode, backward=backward, scaling=scaling, global_context=global_context).cuda()
net.set_step_size(dt)

lr=1e-4
optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200*len(train_dataloader), gamma=0.3)

step = 0
for epoch in range(300):
    for idx, imgs in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        img_loss, smt_loss = 0, 0
        F_YXs = []
        if mode == '3d':
            ref_imgs = imgs.cuda()
            X, Y = ref_imgs[:, 0:1], ref_imgs[:, 1:2]

            flows = net(torch.cat([X, Y], 1))
            flows_YX = [f[:, :3] for f in flows]
            F_YX = utils.ResizeTransform(flows_YX[-1], image_size, 3)
            pred_YX = stn(Y, F_YX)
            img_loss += ncc(X, pred_YX)
            smt_loss += smooth_loss(F_YX) 
            loss = img_loss + smt_loss
            F_YXs.append(F_YX.cpu().detach())
        else:
            if np.random.rand() < 0.5:
                ref_imgs = imgs[:, order1]
            else:
                ref_imgs = imgs[:, order2]
            ref_imgs = ref_imgs.cuda()
            flows = net(ref_imgs)

            flows_YX = [f[:, :3] for f in flows]
            
            YXs = []
            img_loss, smt_loss = 0, 0
            
            for i in range(1, len(flows)):
                F_YX = utils.ResizeTransform(flows_YX[i], image_size, 3)
                smt_loss += smooth_loss(F_YX) / len(flows[1:])
                
                X, Y = ref_imgs[:, 0:1], ref_imgs[:, i:i+1]

                pred_YX = stn(Y, F_YX)
                img_loss += ncc(X, pred_YX) / 5
                F_YXs.append(F_YX.cpu().detach())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 2)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        if step % log_interval == 0:
            log_writer.add_scalar('img_loss', img_loss.item(), step)
            log_writer.add_scalar('smooth_loss', smt_loss.item(), step)

        if step % vis_interval == 0:
            utils.visualize_3d_flow(log_writer, step, 'YX', F_YXs)
            utils.visualize_3d_ground_truth(log_writer, step, imgs, [0, 1])

        if step % checkpoint_interval:
            torch.save(net.state_dict(), f'checkpoints/{task_name}/{step}.pth')

        step += 1


