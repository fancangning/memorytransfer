import numpy as np
import os
import sys

from numpy.lib.utils import source
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.utils as v_utils
import matplotlib.pyplot as plt
import cv2
import math
from collections import OrderedDict
import copy
import time
from sklearn.metrics import roc_auc_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def distribution(source_statistics, target_statistics, labels, log_dir):
    # source_statistics, target_statistics and labels are all list of tensors
    source_statistics = torch.cat(source_statistics, 0).detach().cpu()
    target_statistics = torch.cat(target_statistics, 0).detach().cpu()
    print('source_statistics size:', source_statistics.size())
    print('target_statistics size:', target_statistics.size())
    print('labels size:', labels.size())
    assert(target_statistics.size()==labels.size())

    target_normal_mask = labels.eq(0)
    target_abnormal_mask = labels.eq(1)

    target_normal_statistics = torch.masked_select(target_statistics, target_normal_mask)
    target_abnormal_statistics = torch.masked_select(target_statistics, target_abnormal_mask)

    max_length = max(len(target_normal_statistics), len(target_abnormal_statistics), len(source_statistics))

    if len(target_normal_statistics) == max_length:
        source_statistics = torch.cat([source_statistics.cpu(), torch.tensor([np.nan] * (max_length - len(source_statistics)))])
        target_abnormal_statistics = torch.cat([target_abnormal_statistics.cpu(), torch.tensor([np.nan] * (max_length - len(target_abnormal_statistics)))])
    elif len(target_abnormal_statistics) == max_length:
        source_statistics = torch.cat([source_statistics.cpu(), torch.tensor([np.nan] * (max_length - len(source_statistics)))])
        target_normal_statistics = torch.cat([target_normal_statistics.cpu(), torch.tensor([np.nan] * (max_length - len(target_normal_statistics)))])
    elif len(source_statistics) == max_length:
        target_normal_statistics = torch.cat([target_normal_statistics.cpu(), torch.tensor([np.nan] * (max_length - len(target_normal_statistics)))])
        target_abnormal_statistics = torch.cat([target_abnormal_statistics.cpu(), torch.tensor([np.nan] * (max_length - len(target_abnormal_statistics)))])
    else:
        raise Exception('wrong length')
    
    df = pd.DataFrame(
        {
            'target_normal_statistics': target_normal_statistics,
            'target_abnormal_statistics': target_abnormal_statistics,
            'source_statistics': source_statistics,
        }
    )
    fig, ax = plt.subplots()
    sns.kdeplot(data=df, fill=True, common_norm=False, palette='crest', alpha=.5, linewidth=0)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fig.savefig(os.path.join(log_dir, 'pretrain_statistics_kde.png'))
    return fig

def weight_func(x, scaling=1e5, offset=0.6):
    m = nn.Sigmoid()
    x = scaling*(x - offset)
    return m(x).detach()

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def psnr(mse):

    return 10 * math.log10(1 / mse)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def normalize_img(img):

    img_re = copy.copy(img)
    
    img_re = (img_re - np.min(img_re)) / (np.max(img_re) - np.min(img_re))
    
    return img_re

def point_score(outputs, imgs):
    
    loss_func_mse = nn.MSELoss(reduction='none')
    error = loss_func_mse((outputs[0]+1)/2,(imgs[0]+1)/2)
    normal = (1-torch.exp(-error)) # range [0, 1)
    score = (torch.sum(normal*loss_func_mse((outputs[0]+1)/2,(imgs[0]+1)/2)) / torch.sum(normal)).item()
    return score
    
def anomaly_score(psnr, max_psnr, min_psnr):
    return ((psnr - min_psnr) / (max_psnr-min_psnr))

def anomaly_score_inv(psnr, max_psnr, min_psnr):
    return (1.0 - ((psnr - min_psnr) / (max_psnr-min_psnr)))

def anomaly_score_list(psnr_list):
    anomaly_score_list = list()
    for i in range(len(psnr_list)):
        anomaly_score_list.append(anomaly_score(psnr_list[i], np.max(psnr_list), np.min(psnr_list)))
        
    return anomaly_score_list

def anomaly_score_list_inv(psnr_list):
    anomaly_score_list = list()
    for i in range(len(psnr_list)):
        anomaly_score_list.append(anomaly_score_inv(psnr_list[i], np.max(psnr_list), np.min(psnr_list)))
        
    return anomaly_score_list

def AUC(anomal_scores, labels):
    frame_auc = roc_auc_score(y_true=np.squeeze(labels, axis=0), y_score=np.squeeze(anomal_scores))
    return frame_auc

def score_sum(list1, list2, alpha):
    list_result = []
    assert len(list1) == len(list2)
    for i in range(len(list1)):
        list_result.append((alpha*list1[i]+(1-alpha)*list2[i]))
        
    return list_result
