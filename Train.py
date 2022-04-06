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
from sklearn.metrics import roc_auc_score
from utils import *
import random

import argparse

torch.multiprocessing.set_sharing_strategy('file_system')


parser = argparse.ArgumentParser(description="memorytransfer")
parser.add_argument('--gpus', nargs='+', type=str, help='gpus')
parser.add_argument('--batch_size', type=int, default=4, help='batch size for training')
parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs for training')
parser.add_argument('--loss_similarity', type=float, default=0.1, help='weight of the memory similarity loss')
parser.add_argument('--loss_diversity', type=float, default=0.1, help='weight of the memory diversity loss')
parser.add_argument('--loss_adversarial', type=float, default=0.1, help='weight of the adversarial loss')
parser.add_argument('--scaling', type=float, default=1e5, help='the scaling of weight function')
parser.add_argument('--offset', type=float, default=0.6, help='the offset of weight function')
parser.add_argument('--h', type=int, default=256, help='height of input images')
parser.add_argument('--w', type=int, default=256, help='width of input images')
parser.add_argument('--c', type=int, default=3, help='channel of input images')
parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
parser.add_argument('--t_length', type=int, default=5, help='length of the frame sequences')
parser.add_argument('--fdim', type=int, default=512, help='channel dimension of the features')
parser.add_argument('--mdim', type=int, default=512, help='channel dimension of the memory items')
parser.add_argument('--msize', type=int, default=10, help='number of the memory items')
parser.add_argument('--num_workers', type=int, default=2, help='number of workers for the train loader')
parser.add_argument('--num_workers_test', type=int, default=1, help='number of workers for the test loader')
parser.add_argument('--source_dataset_type', type=str, default='ped1', help='type of dataset: ped1, ped2')
parser.add_argument('--target_dataset_type', type=str, default='ped2', help='type of dataset: ped1, ped2')
parser.add_argument('--dataset_path', type=str, default='../data', help='directory of data')
parser.add_argument('--exp_dir', type=str, default='log', help='directory of log')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# args.gpus is a list
if args.gpus is None:
	gpus = "0"
	os.environ["CUDA_VISIBLE_DEVICES"]= gpus
else:
	gpus = ""
	for i in range(len(args.gpus)):
		gpus = gpus + args.gpus[i] + ","
	os.environ["CUDA_VISIBLE_DEVICES"]= gpus[:-1]

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

source_train_folder = os.path.join(args.dataset_path, args.source_dataset_type, 'training', 'frames')
target_train_folder = os.path.join(args.dataset_path, args.target_dataset_type, 'testing', 'frames')
test_folder = os.path.join(args.dataset_path, args.target_dataset_type, 'testing', 'frames')
pretrain_folder = os.path.join(args.dataset_path, args.source_dataset_type, 'training', 'frames')

# Report the training process
report_start_time = time.time()

log_dir = os.path.join('./exp', args.source_dataset_type+'_to_'+args.target_dataset_type, args.method, args.exp_dir)
if not os.path.exists(log_dir):
	os.makedirs(log_dir)
orig_stdout = sys.stdout
f = open(os.path.join(log_dir, 'log.txt'),'w')
sys.stdout= f
print('--------------------------------------Parameters Setting--------------------------------------')
for arg in vars(args):
	print(arg, getattr(args, arg))
print('----------------------------------------------------------------------------------------------')

# Loading pretrain dataset
loading_start_time = time.time()

pretrain_dataset = DataLoader(pretrain_folder, transforms.Compose([
			transforms.ToTensor(),
			]), resize_height=args.h, resize_width=args.w, time_step=args.t_length-1)

# source_train_size = len(source_train_dataset)
# target_train_size = len(target_train_dataset)
# test_size = len(test_dataset)

pretrain_batch = data.DataLoader(pretrain_dataset, batch_size= args.batch_size,
							 shuffle=True, num_workers=args.num_workers, drop_last=True)

pretrain_batch_length = len(list(pretrain_batch))

# source_batch_length = sum(1 for _ in source_train_batch)
# target_batch_length = sum(1 for _ in target_train_batch)
# pretrain_batch_length = sum(1 for _ in pretrain_batch)

# Model setting
model_setting_start_time = time.time()

from model.final_future_prediction_with_memory_spatial_sumonly_weight_ranking_top1 import *
model = convAE(args.c, args.t_length, args.msize, args.fdim, args.mdim)

params_encoder = list(model.encoder.parameters())
params_decoder = list(model.decoder.parameters())

pretrain_optimizer = torch.optim.Adam(params_encoder+params_decoder, lr=args.lr)
pretrain_scheduler = optim.lr_scheduler.CosineAnnealingLR(pretrain_optimizer, T_max=args.epochs)

model.cuda()

# Training
training_start_time = time.time()

loss_func_mse = nn.MSELoss(reduction='none')
m_items = F.normalize(torch.rand((args.msize, args.mdim), dtype=torch.float), dim=1).cuda() # Initialize the memory items

# Pretraining
if os.path.exists(os.path.join('.', 'pretrain_model.pth')) and os.path.exists(os.path.join('.', 'pretrain_keys.pt')):
	print('pretrained model is found. load the pretrained model.')
	# Loading the trained model
	model = torch.load(os.path.join('.', 'pretrain_model.pth'))
	m_items = torch.load(os.path.join('.', 'pretrain_keys.pt'))

else:
	print('pretrained model is not found. begin to pretrain the model.')
	model.train()
	for epoch in range(args.epochs):

		total_pretrain_loss_pixel = 0
		total_pretrain_diversity_loss = 0
		total_pretrain_similarity_loss = 0
		for idx, imgs in enumerate(pretrain_batch):
			imgs = imgs.cuda()

			outputs, fea, updated_fea, fea_cat, m_items, softmax_score_query, softmax_score_memory, diversity_loss, similarity_loss = model.forward(imgs[:,0:12], m_items, None, True)
			
			pretrain_optimizer.zero_grad()
			loss_pixel = torch.mean(loss_func_mse(outputs, imgs[:,12:]))
			
			loss = loss_pixel + args.loss_diversity * diversity_loss + args.loss_similarity * similarity_loss
			loss.backward()
			pretrain_optimizer.step()

			total_pretrain_loss_pixel += loss_pixel.item()
			total_pretrain_diversity_loss += diversity_loss.item()
			total_pretrain_similarity_loss += similarity_loss.item()
		pretrain_scheduler.step()

	torch.save(model, os.path.join(log_dir, 'pretrain_model.pth'))
	torch.save(m_items, os.path.join(log_dir, 'pretrain_keys.pt'))

# the pretrain stage ends, the train stage begins
source_train_dataset = DataLoader(source_train_folder, transforms.Compose([
			 transforms.ToTensor(),          
			 ]), resize_height=args.h, resize_width=args.w, time_step=args.t_length-1)

target_train_dataset = DataLoader(target_train_folder, transforms.Compose([
			 transforms.ToTensor(),
			 ]), resize_height=args.h, resize_width=args.w, time_step=args.t_length-1, alpha=0.6, model=model, m_items_test=m_items.clone(), is_anomaly_score=True)

test_dataset = DataLoader(test_folder, transforms.Compose([
			 transforms.ToTensor(),            
			 ]), resize_height=args.h, resize_width=args.w, time_step=args.t_length-1)

source_train_batch = data.DataLoader(source_train_dataset, batch_size = args.batch_size, 
							  shuffle=True, num_workers = args.num_workers, drop_last = True)
target_train_batch = data.DataLoader(target_train_dataset, batch_size = args.batch_size,
							  shuffle=True, num_workers = args.num_workers, drop_last = True)
test_batch = data.DataLoader(test_dataset, batch_size = args.test_batch_size, 
							 shuffle=False, num_workers=args.num_workers_test, drop_last=False)

source_train_batch = list(source_train_batch)
target_train_batch = list(target_train_batch)

source_batch_length = len(source_train_batch)
target_batch_length = len(target_train_batch)

from model.final_future_prediction_with_memory_spatial_sumonly_weight_ranking_top1 import *
model = convAE(args.c, args.t_length, args.msize, args.fdim, args.mdim)

params_encoder = list(model.encoder.parameters())
params_decoder = list(model.decoder.parameters())

model.cuda()

m_items = F.normalize(torch.rand((args.msize, args.mdim), dtype=torch.float), dim=1).cuda() # Initialize the memory items

discriminator = DomainDiscriminator(in_feature=args.fdim*2, hidden_size=args.fdim)
discriminator_rGradient = DomainDiscriminator(in_feature=args.fdim, hidden_size=int(args.fdim/2))
grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=args.epochs*max(source_batch_length, target_batch_length), auto_step=True)
params_discriminator = list(discriminator.parameters())
params_discriminator_rGradient = list(discriminator_rGradient.parameters())
params = params_encoder + params_decoder + params_discriminator + params_discriminator_rGradient

optimizer = torch.optim.Adam(params, lr=args.lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max =args.epochs)

discriminator.cuda()
discriminator_rGradient.cuda()

for epoch in range(args.epochs):
	model.train()
	discriminator.train()
	discriminator_rGradient.train()
	
	start = time.time()

	total_loss_pixel = 0
	total_similarity_loss = 0
	total_diversity_loss = 0
	total_adversarial_loss = 0
	total_adversarial_loss_rGradient = 0

	for idx in range(max(source_batch_length, target_batch_length)):
		s_idx = idx % source_batch_length
		t_idx = idx % target_batch_length
		
		s_imgs = source_train_batch[s_idx]
		t_imgs, t_anomaly_score = target_train_batch[t_idx]

		# print('The size of imgs:', imgs.size()) 8x15x256x256
		imgs = torch.cat((s_imgs, t_imgs), 0)
		imgs = imgs.cuda()

		outputs, fea, updated_fea, fea_cat, m_items, softmax_score_query, softmax_score_memory, diversity_loss, similarity_loss = model.forward(imgs[:,0:12], m_items, None, None)

		optimizer.zero_grad()
		# calculate the distance between frame and memory item

		# the distance between every query and its closest memory item
		# distance_query_memory, _ = torch.min(softmax_score_memory, -1)
		# distance_query_memory = distance_query_memory.reshape(2*args.batch_size, -1)

		# the definition of the distance between the frame and memory item
		# distance_frame_memory, _ = torch.max(distance_query_memory, -1)
		# distance_frame_memory_target = distance_frame_memory[args.batch_size:]
		# distance_frame_memory_target = distance_frame_memory_target / torch.max(distance_frame_memory_target)

		# loss_pixel = torch.mean(loss_func_mse(outputs[:args.batch_size], imgs[:args.batch_size,12:])) + torch.mean(distance_frame_memory_target*torch.mean(loss_func_mse(outputs[args.batch_size:], imgs[args.batch_size:,12:]).reshape(args.batch_size, -1), -1))
		
		# target_recon_loss = torch.mean(loss_func_mse(outputs[args.batch_size:], imgs[args.batch_size:,12:]).reshape(args.batch_size, -1), 1).detach()
		# recon_loss_weight = weight_func(target_recon_loss, scaling=args.scaling, offset=args.offset)
		# loss_pixel = torch.mean(loss_func_mse(outputs[:args.batch_size], imgs[:args.batch_size,12:])) + torch.mean(recon_loss_weight*torch.mean(loss_func_mse(outputs[args.batch_size:], imgs[args.batch_size:,12:]).reshape(args.batch_size, -1), -1))
		
		# loss_pixel = torch.mean(loss_func_mse(outputs[:args.batch_size], imgs[:args.batch_size,12:]))

		recon_loss_weight = weight_func(t_anomaly_score.cuda(), scaling=args.scaling, offset=args.offset)
		loss_pixel = torch.mean(loss_func_mse(outputs[:args.batch_size], imgs[:args.batch_size,12:])) + torch.mean(recon_loss_weight*torch.mean(loss_func_mse(outputs[args.batch_size:], imgs[args.batch_size:,12:]).reshape(args.batch_size, -1), -1))
		
		if epoch % 10 == 9 and idx == max(source_batch_length, target_batch_length) - 1:
			original_reconstruction_imgs = torch.cat((imgs[:,12:], outputs), 0)
			img_grid = torchvision.utils.make_grid(original_reconstruction_imgs, 2*args.batch_size)
			writer.add_image('imgs/epoch_'+str(epoch), img_grid)
		# fea 8x512x32x32
		fea = fea.permute(0, 2, 3, 1)
		fea = fea.reshape(-1, args.fdim)
		fea = grl(fea)
		domain_output = discriminator_rGradient(fea)
		s_domain_output, t_domain_output = domain_output.chunk(2, dim=0)
		s_domain_label = torch.zeros((s_domain_output.size(0), 1)).cuda()
		t_domain_label = torch.ones((t_domain_output.size(0), 1)).cuda()
		adversarial_loss_rGradient = 0.5 * (nn.BCELoss()(s_domain_output, s_domain_label) + nn.BCELoss()(t_domain_output, t_domain_label))

		# fea_cat 8x1024x32x32
		fea_cat = fea_cat.clone().detach()
		fea_cat = fea_cat.permute(0, 2, 3, 1)
		fea_cat = fea_cat.reshape(-1, args.fdim*2)
		domain_output = discriminator(fea_cat)
		s_domain_output, t_domain_output = domain_output.chunk(2, dim=0)
		s_domain_label = torch.zeros((s_domain_output.size(0), 1)).cuda()
		t_domain_label = torch.ones((t_domain_output.size(0), 1)).cuda()
		adversarial_loss = 0.5 * (nn.BCELoss()(s_domain_output, s_domain_label) + nn.BCELoss()(t_domain_output, t_domain_label))
		
		loss = loss_pixel + args.loss_similarity * similarity_loss + args.loss_diversity * diversity_loss + args.loss_adversarial * (adversarial_loss + adversarial_loss_rGradient)
		# loss = loss_pixel + args.loss_similarity * similarity_loss + args.loss_diversity * diversity_loss
		loss.backward()
		optimizer.step()

		total_loss_pixel += loss_pixel.item()
		total_similarity_loss += similarity_loss.item()
		total_diversity_loss += diversity_loss.item()
		total_adversarial_loss += adversarial_loss.item()
		total_adversarial_loss_rGradient += adversarial_loss_rGradient.item()


	if epoch % 10 == 9:
		torch.save(model, os.path.join(log_dir, 'model_'+str(epoch)+'.pth'))
		torch.save(discriminator, os.path.join(log_dir, 'discriminator_'+str(epoch)+'.pth'))
		torch.save(discriminator_rGradient, os.path.join(log_dir, 'discriminator_rGradient_'+str(epoch)+'.pth'))
		torch.save(m_items, os.path.join(log_dir, 'keys_'+str(epoch)+'.pt'))
		
	scheduler.step()
	
	print('----------------------------------------')
	print('Epoch:', epoch+1)
	if args.method == 'pred':
		print('Loss: Prediction {:.6f}/ Similarity {:.6f}/ Diversity {:.6f}/ Adversarial {:.6f}/ Adversarial_rGradient {:.6f}'.format(loss_pixel.item(), similarity_loss.item(), diversity_loss.item(), adversarial_loss.item(), adversarial_loss_rGradient.item()))
	else:
		print('Loss: Reconstruction {:.6f}/ Compactness {:.6f}/ Separateness {:.6f}/ Adversarial {:.6f}'.format(loss_pixel.item(), similarity_loss.item(), diversity_loss.item(), adversarial_loss.item()))
	print('----------------------------------------')
	
print('Training is finished')
# Save the model and the memory items
saving_start_time = time.time()

torch.save(model, os.path.join(log_dir, 'model.pth'))
torch.save(discriminator, os.path.join(log_dir, 'discriminator.pth'))
torch.save(discriminator_rGradient, os.path.join(log_dir, 'discriminator_rGradient.pth'))
torch.save(m_items, os.path.join(log_dir, 'keys.pt'))

end_time = time.time()
print('report time:', loading_start_time-report_start_time)
print('loading time:', model_setting_start_time-loading_start_time)
print('model setting time:', training_start_time-model_setting_start_time)
print('training time per epoch:', (saving_start_time-training_start_time)/args.epochs * 2)
print('saving time:', end_time-saving_start_time)
sys.stdout = orig_stdout
f.close()
