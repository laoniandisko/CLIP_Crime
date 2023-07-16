import os
from PIL import Image
import numpy as np
import torch
import clip
from loguru import logger
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
from tqdm import tqdm
from dataset import YourDataset

if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net, preprocess = clip.load("RN50",device=device,jit=False)

    optimizer = optim.Adam(net.parameters(), lr=1e-6,betas=(0.9,0.98),eps=1e-6,weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.1)

    # 创建损失函数
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()


    your_dataset = YourDataset(img_root= '../archive',meta_root= '../archive',is_train=True,preprocess=preprocess)
    dataset_size_your = len(your_dataset)
    your_dataloader = DataLoader(your_dataset,batch_size=4,shuffle=True,num_workers=4,pin_memory=False)


    phase = "train"
    model_name = "CLIP_Crime"
    ckt_gap = 4
    epoches = 30
    for epoch in range(epoches):
        scheduler.step()
        total_loss = 0
        batch_num = 0
        # 使用混合精度，占用显存更小
        with torch.cuda.amp.autocast(enabled=True):
            progress_bar = tqdm(total=len(your_dataloader))
            for images,label_tokens in tqdm(your_dataloader) :
                # 将图片和标签token转移到device设备
                images = images.to(device)
                label_tokens = label_tokens.to(device)
                batch_num += 1
                # 优化器梯度清零
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    logits_per_image, logits_per_text = net(images, label_tokens)
                    ground_truth = torch.arange(len(images),dtype=torch.long,device=device)
                    cur_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
                    total_loss += cur_loss
                    if phase == "train":
                        cur_loss.backward()
                        if device == "cpu":
                            optimizer.step()
                        else:
                            optimizer.step()
                            clip.model.convert_weights(net)
                if batch_num % 4 == 0:
                    progress_bar.write('{} epoch:{} loss:{}'.format(phase,epoch,cur_loss))
                    logger.info('{} epoch:{} loss:{}'.format(phase,epoch,cur_loss))
                progress_bar.update(1)
            epoch_loss = total_loss / dataset_size_your
            torch.save(net.state_dict(),f"{model_name}_epoch_{epoch}.pth")
            logger.info(f"weights_{epoch} saved")
            if epoch % ckt_gap == 0:
                checkpoint_path = f"{model_name}_ckt.pth"
                checkpoint = {
                    'it': epoch,
                    'network': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()}
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"checkpoint_{epoch} saved")
            logger.info('{} Loss: {:.4f}'.format(
                phase, epoch_loss))

