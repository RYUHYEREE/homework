
import pickle
import argparse
import yaml
import pickle
import argparse
import yaml
import numpy as np
import albumentations as albu
from collections import OrderedDict
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.net import EncoderDecoderNet, SPPNet
from losses.multi import MultiClassCriterion
from logger.log import debug_logger
from logger.plot import history_ploter
from utils.optimizer import create_optimizer
from utils.metrics import compute_iou_batch


parser = argparse.ArgumentParser()
parser.add_argument('config_path')
parser.add_argument('-device_id', default='0')
args = parser.parse_args()
config_path = Path(args.config_path)
config = yaml.load(open(config_path))
net_config = config['Net']
data_config = config['Data']
train_config = config['Train']
loss_config = config['Loss']
opt_config = config['Optimizer']
device = torch.device('cuda:'+args.device_id if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)
t_max = opt_config['t_max']

max_epoch = train_config['max_epoch']
batch_size = train_config['batch_size']
fp16 = train_config['fp16']
resume = train_config['resume']
pretrained_path = train_config['pretrained_path']


    #deeplab
net_type = 'deeplab'
model = SPPNet(**net_config)#딱 한번 CNN을 실행함
                                #이미지 크기에 상관이 없음
                                #feature map을 한번만 계산
                                #다중수준의 pooling으로 object detection에 좋음

dataset = data_config['dataset']
    #bdd100k붙은것만 확인 
if dataset == 'bdd100k':
    from dataset.bdd100k import BDD100KDataset as Dataset
    net_config['output_channels'] = 3 
    classes = np.arange(1, 3)
elif dataset == 'bdd100k2':
    from dataset.bdd100k2 import BDD100K2Dataset as Dataset
    net_config['output_channels'] = 4
    classes = np.arange(1, 4)
elif dataset == 'bdd100k3':
    from dataset.bdd100k3 import BDD100K3Dataset as Dataset
    net_config['output_channels'] = 7 
    classes = np.arange(1, 7)
    print(classes)
elif dataset == 'bdd100k4':
    from dataset.bdd100k4 import BDD100K4Dataset as Dataset
    net_config['output_channels'] = 12 
    classes = np.arange(1, 12) 
elif dataset == 'bdd100k5':
    from dataset.bdd100k5 import BDD100K5Dataset as Dataset
    net_config['output_channels'] = 10
    classes = np.arange(1,10)
else:
    raise NotImplementedError
del data_config['dataset']

modelname = config_path.stem
output_dir = Path('../model') / modelname
output_dir.mkdir(exist_ok=True)
log_dir = Path('../logs') / modelname
log_dir.mkdir(exist_ok=True)

logger = debug_logger(log_dir)
logger.debug(config)
logger.info(f'Device: {device}')
logger.info(f'Max Epoch: {max_epoch}')

# Loss
loss_fn = MultiClassCriterion(**loss_config).to(device)#다중클래스 손실함수
params = model.parameters()
optimizer, scheduler = create_optimizer(params, **opt_config)

# history
if resume:
    with open(log_dir.joinpath('history.pkl'), 'rb') as f:
        history_dict = pickle.load(f)
        best_metrics = history_dict['best_metrics']
        loss_history = history_dict['loss']
        iou_history = history_dict['iou']
        start_epoch = len(iou_history)
        for _ in range(start_epoch):
            scheduler.step()
else:
    start_epoch = 0
    best_metrics = 0
    loss_history = []
    iou_history = []

# Dataset
#이미지의 데이터를 부풀려서 성능을 좋게 만드는 라이브러리 albumentation
affine_augmenter = albu.Compose([albu.HorizontalFlip(p=.5),
                             # Rotate(5, p=.5)
                             ])
# image_augmenter = albu.Compose([albu.GaussNoise(p=.5),
#                                 albu.RandomBrightnessContrast(p=.5)])
image_augmenter = None
#print(data_config)
#데이터 셋을 가져옴
train_dataset = Dataset(affine_augmenter=affine_augmenter, image_augmenter=image_augmenter,
                    net_type=net_type, **data_config)
#valid_dataset = Dataset(split='valid', net_type=net_type, **data_config)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                          pin_memory=True, drop_last=True)
#valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

# To device
#device cpu에서 사용할 수 있게 해주는거
model = model.to(device)

# Pretrained model
if pretrained_path:
    logger.info(f'Resume from {pretrained_path}')
    param = torch.load(pretrained_path)
    model.load_state_dict(param)
    del param

# fp16
#부동소수점 연산(16비트)
#텐서를 반으로 잘라서 연산해서 연산 시간이 줄어든
if fp16:
    from apex import fp16_utils
    model = fp16_utils.BN_convert_float(model.half())
    optimizer = fp16_utils.FP16_Optimizer(optimizer, verbose=False, dynamic_loss_scale=True)
    logger.info('Apply fp16')

# Restore model
if resume:
    model_path = output_dir.joinpath(f'model_tmp.pth')
    logger.info(f'Resume from {model_path}')
    param = torch.load(model_path)
    model.load_state_dict(param)
    del param
    opt_path = output_dir.joinpath(f'opt_tmp.pth')
    param = torch.load(opt_path)
    optimizer.load_state_dict(param)
    del param

# 학습 
for i_epoch in range(start_epoch, max_epoch):
    logger.info(f'Epoch: {i_epoch}')
    logger.info(f'Learning rate: {optimizer.param_groups[0]["lr"]}')

    train_losses = []
    train_ious = []
    model.train()
    #상태바 
    with tqdm(train_loader) as _tqdm:
        for batched in _tqdm:
            images, labels = batched
            if fp16:
                images = images.half()
            images, labels = images.to(device), labels.to(device)
            #print("Lables backed from gpu : ", labels.size()) # --> [0, 1, 2] or [0, 1]
            optimizer.zero_grad()
            preds = model(images)
            #deeplab 시멘틱 세그멘테이션
            #입력 영상에 주어진 각각의 픽셀에 대해서 class label을 할당하는 것
            if net_type == 'deeplab':
                preds = F.interpolate(preds, size=labels.shape[1:], mode='bilinear', align_corners=True)
                #보간
            if fp16:
                loss = loss_fn(preds.float(), labels)
            else:
                loss = loss_fn(preds, labels)
            #연산을 cpu에서 함! cpu에서 연산 끝나고 뺌
            #라벨은 그림에서 그리기로 했던 것들 0-12까지 뜸
            preds_np = preds.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            print("Preds", np.unique(np.argmax(preds_np, axis=1)))
            print("Labels", np.unique(labels_np))
            #YOLO보다 좋은 알고리즘
            #얼마나 예측한 값이랑 실제 값이 오차가 적은지에 대한 행렬(오차?)
            iou = compute_iou_batch(np.argmax(preds_np, axis=1), labels_np, classes)
            
            _tqdm.set_postfix(OrderedDict(seg_loss=f'{loss.item():.5f}', iou=f'{iou:.3f}'))
            train_losses.append(loss.item())
            train_ious.append(iou)

            if fp16:
                optimizer.backward(loss)
            else:
                loss.backward()
            optimizer.step()

    #최적화를 더 좋게 최적화하는걸 한단계 밟아
    scheduler.step()

    train_loss = np.mean(train_losses)
    train_iou = np.nanmean(train_ious)
    logger.info(f'train loss: {train_loss}')
    logger.info(f'train iou: {train_iou}')

    torch.save(model.state_dict(), output_dir.joinpath('model_tmp.pth'))
    torch.save(optimizer.state_dict(), output_dir.joinpath('opt_tmp.pth'))

#    if (i_epoch + 1) % 10 == 0:
#        valid_losses = []
#        valid_ious = []
#        #평가해준다(모델을 평가해)
#        model.eval()
#        with torch.no_grad():
#            with tqdm(valid_loader) as _tqdm:
#                for batched in _tqdm:
#                    images, labels = batched
#                    if fp16:
#                        images = images.half()
#                    images, labels = images.to(device), labels.to(device)
#                    preds = model.tta(images, net_type=net_type)
#                    if fp16:
#                        loss = loss_fn(preds.float(), labels)
#                    else:
#                        loss = loss_fn(preds, labels)
#
#                    preds_np = preds.detach().cpu().numpy()
#                    labels_np = labels.detach().cpu().numpy()
#                    iou = compute_iou_batch(np.argmax(preds_np, axis=1), labels_np, classes)
#                    valid_losses.append(loss.item())
#                    valid_ious.append(iou)
#
#        valid_loss = np.mean(valid_losses)
#        valid_iou = np.mean(valid_ious)
#        logger.info(f'valid seg loss: {valid_loss}')
#        logger.info(f'valid iou: {valid_iou}')
#
#        #iou는 값이 클수록 좋은거임
#        #왜냐하면 전체부분중에 일치하는 부분의 영역이니까
#        if best_metrics < valid_iou:
#            best_metrics = valid_iou
#            logger.info('Best Model!')
#            torch.save(model.state_dict(), output_dir.joinpath('model.pth'))
#            torch.save(optimizer.state_dict(), output_dir.joinpath('opt.pth'))
#    else:
#        valid_loss = None
#        valid_iou = None

#    loss_history.append([train_loss, valid_loss])
#    iou_history.append([train_iou, valid_iou])
#    history_ploter(loss_history, log_dir.joinpath('loss.png'))
#    history_ploter(iou_history, log_dir.joinpath('iou.png'))

#    history_dict = {'loss': loss_history,
#                    'iou': iou_history,
#                    'best_metrics': best_metrics}
#    with open(log_dir.joinpath('history.pkl'), 'wb') as f:
#        pickle.dump(history_dict, f)

