import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.nn.init as init
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.utils as v_utils
import matplotlib.pyplot as plt
import cv2
import math
from collections import OrderedDict
import copy
import time
from model.utils import DataLoader
from model import utils
from model.final_future_prediction_with_memory_spatial_sumonly_weight_ranking_top1 import *
from sklearn import metrics
from utils import *
import random
import glob

import argparse


parser = argparse.ArgumentParser(description="evaluate the performance of a specific model")
parser.add_argument('--gpus', nargs='+', type=str, help='gpus')
parser.add_argument('--batch_size', type=int, default=4, help='batch size for training')
parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
parser.add_argument('--h', type=int, default=256, help='height of input images')
parser.add_argument('--w', type=int, default=256, help='width of input images')
parser.add_argument('--c', type=int, default=3, help='channel of input images')
parser.add_argument('--t_length', type=int, default=5, help='length of the frame sequences')
parser.add_argument('--fdim', type=int, default=512, help='channel dimension of the features')
parser.add_argument('--mdim', type=int, default=512, help='channel dimension of the memory items')
parser.add_argument('--msize', type=int, default=10, help='number of the memory items')
parser.add_argument('--alpha', type=float, default=0.6, help='weight for the abnormality score')
parser.add_argument('--th', type=float, default=0.01, help='threshold for test updating')
parser.add_argument('--num_workers', type=int, default=2, help='number of workers for the train loader')
parser.add_argument('--num_workers_test', type=int, default=1, help='number of workers for the test loader')
parser.add_argument('--test_dataset_type', type=str, default='ped2', help='type of dataset: ped2, ped1')
parser.add_argument('--data_path', type=str, default='../data', help='the path of the test dataset')
parser.add_argument('--model_path', type=str, default='./exp/ped1_to_ped2/pred/log_gpu1_1006_1225', help='directory of model')
parser.add_argument('--model_name', type=str, default='pretrain_model.pth', help='name of model')
parser.add_argument('--m_items_name', type=str, default='pretrain_keys.pt', help='name of m_items')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
if args.gpus is None:
    gpus = "0"
    os.environ["CUDA_VISIBLE_DEVICES"]= gpus
else:
    gpus = ""
    for i in range(len(args.gpus)):
        gpus = gpus + args.gpus[i] + ","
    os.environ["CUDA_VISIBLE_DEVICES"]= gpus[:-1]

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

test_folder = os.path.join(args.data_path, args.test_dataset_type, 'testing', 'frames')

# Loading dataset
test_dataset = DataLoader(test_folder, transforms.Compose([
             transforms.ToTensor(),            
             ]), resize_height=args.h, resize_width=args.w, time_step=args.t_length-1)

# test_size = len(test_dataset)

test_batch = data.DataLoader(test_dataset, batch_size = args.test_batch_size, 
                             shuffle=False, num_workers=args.num_workers_test, drop_last=False)

loss_func_mse = nn.MSELoss(reduction='none')

# Loading the trained model
model = torch.load(os.path.join(args.model_path, args.model_name))
model.cuda()
m_items = torch.load(os.path.join(args.model_path, args.m_items_name))
m_items.cuda()
labels = np.load(os.path.join(args.data_path, 'frame_labels_'+args.test_dataset_type+'.npy'))

videos = OrderedDict()
videos_list = sorted(glob.glob(os.path.join(test_folder, '*')))
for video in videos_list:
    video_name = video.split('/')[-1] # e.g. 01 02 03
    videos[video_name] = {}
    videos[video_name]['path'] = video # the relative path of video
    videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))
    videos[video_name]['frame'].sort()
    videos[video_name]['length'] = len(videos[video_name]['frame'])

labels_list = []
label_length = 0
# peak signal noise ratio
psnr_list = {} 
feature_distance_list = {}

print('Evaluation of', args.test_dataset_type)

# Setting for video anomaly detection
for video in sorted(videos_list):
    video_name = video.split('/')[-1] # e.g. 01 02 03
    labels_list = np.append(labels_list, labels[0][4+label_length:videos[video_name]['length']+label_length])
    label_length += videos[video_name]['length']
    psnr_list[video_name] = list()
    feature_distance_list[video_name] = list()

label_length = 0
video_num = 0
label_length += videos[videos_list[video_num].split('/')[-1]]['length']
m_items_test = m_items.clone()

model.eval()

with torch.no_grad():
    for k, (imgs) in enumerate(test_batch):
        
        # record which video the current frame comes from
        if k == label_length-4*(video_num+1):
            video_num += 1
            label_length += videos[videos_list[video_num].split('/')[-1]]['length']

        # imgs 1x15x256x256
        imgs = Variable(imgs).cuda()
        
        outputs, feas, updated_feas, fea_cat, m_items_test, softmax_score_query, softmax_score_memory, diversity_loss, similarity_loss = model.forward(imgs[:,0:3*4], m_items_test, None, False)
        mse_imgs = torch.mean(loss_func_mse(outputs, imgs[:,12:])).item()
        mse_feas = similarity_loss.item()

        # Calculating the threshold for updating at the test time
        point_sc = point_score(outputs, imgs[:,12:])
        
        
        if point_sc < args.th:
            query = F.normalize(feas, dim=1) # 1X512X32X32
            query = query.permute(0, 2, 3, 1) # 1X32X32X512
            entropy = torch.ones(query.size(0)*query.size(1)*query.size(2), 1).cuda()
            m_items_test, diversity_loss_update = model.memory.update(query, m_items_test, entropy, False)
        
        psnr_list[videos_list[video_num].split('/')[-1]].append(psnr(mse_imgs))
        feature_distance_list[videos_list[video_num].split('/')[-1]].append(mse_feas)

# Measuring the abnormality score and the AUC
anomaly_score_total_list = list()
for video in sorted(videos_list):
    video_name = video.split('/')[-1]
    anomaly_score_total_list += score_sum(anomaly_score_list(psnr_list[video_name]), 
                                anomaly_score_list_inv(feature_distance_list[video_name]), args.alpha)

anomaly_score_total_list = np.asarray(anomaly_score_total_list)

accuracy = AUC(anomaly_score_total_list, np.expand_dims(1-labels_list, 0))

print('The result of ', args.test_dataset_type)
print('AUC: ', accuracy*100, '%')
