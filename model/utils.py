import numpy as np
from collections import OrderedDict
import os
import glob
import cv2
import torch.utils.data as data
import torch
import torch.nn as nn
import math


rng = np.random.RandomState(2020)
def entropy(x):
	epsilon = 0.000001
	return -x*torch.log(x+epsilon)

def psnr(mse):

    return 10 * math.log10(1 / mse)

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

def score_sum(list1, list2, alpha):
    list_result = []
    assert len(list1) == len(list2)
    for i in range(len(list1)):
        list_result.append((alpha*list1[i]+(1-alpha)*list2[i]))
        
    return list_result


def np_load_frame(filename, resize_height, resize_width):
	"""
	Load image path and convert it to numpy.ndarray. Notes that the color channels are BGR and the color space
	is normalized from [0, 255] to [-1, 1].

	:param filename: the full path of image
	:param resize_height: resized height
	:param resize_width: resized width
	:return: numpy.ndarray
	"""
	image_decoded = cv2.imread(filename)
	image_resized = cv2.resize(image_decoded, (resize_width, resize_height))
	image_resized = image_resized.astype(dtype=np.float32)
	image_resized = (image_resized / 127.5) - 1.0
	return image_resized




class DataLoader(data.Dataset):
	def __init__(self, video_folder, transform, resize_height, resize_width, time_step=4, num_pred=1, alpha=0.6, model=None, m_items_test=None, is_anomaly_score=False):
		self.dir = video_folder
		self.transform = transform
		self.videos = OrderedDict()
		self._resize_height = resize_height
		self._resize_width = resize_width
		self._time_step = time_step
		self._num_pred = num_pred
		self.alpha = alpha
		self.model = model
		self.m_items_test = m_items_test
		self.is_anomaly_score = is_anomaly_score
		self.setup()
		self.samples = self.get_all_samples()
		if is_anomaly_score:
			self.anomaly_score = self.get_all_anomaly_score()
			assert len(self.samples) == len(self.anomaly_score)
		
		
	def setup(self):
		# e.g. self.dir ../data/ped2/testing/frames
		videos = glob.glob(os.path.join(self.dir, '*'))
		for video in sorted(videos):
			# e.g. video_name '01'
			video_name = video.split('/')[-1]
			self.videos[video_name] = {}
			self.videos[video_name]['path'] = video
			self.videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))
			self.videos[video_name]['frame'].sort()
			self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])
			
			
	def get_all_samples(self):
		# frames store the paths of frames
		frames = []
		videos = glob.glob(os.path.join(self.dir, '*'))
		# e.g. video ../data/ped2/testing/frames/01
		for video in sorted(videos):
			video_name = video.split('/')[-1]
			for i in range(len(self.videos[video_name]['frame'])-self._time_step):
				frames.append(self.videos[video_name]['frame'][i])
						   
		return frames               
			
	def get_all_anomaly_score(self):
		psnr_list = dict()
		feature_distance_list = dict()

		video_length = 0
		video_num = 0

		videos_list = sorted(glob.glob(os.path.join(self.dir, '*')))
		video_length += self.videos[videos_list[video_num].split('/')[-1]]['length']

		loss_func_mse = nn.MSELoss(reduction='none')
		for video in sorted(videos_list):
			video_name = video.split('/')[-1] # e.g. 01 02 03
			psnr_list[video_name] = list()
			feature_distance_list[video_name] = list()
		
		self.model.eval()

		with torch.no_grad():
			for k in range(len(self.samples)):
				
				# if k > 0:
				# 	exit()
				# record which video the current frame comes from
				if k == video_length-4*(video_num+1):
					video_num += 1
					video_length += self.videos[videos_list[video_num].split('/')[-1]]['length']

				# imgs 1x15x256x256

				index = k
				video_name = self.samples[index].split('/')[-2]
				frame_name = int(self.samples[index].split('/')[-1].split('.')[-2])
				batch = []
				for i in range(self._time_step+self._num_pred):
					image = np_load_frame(self.videos[video_name]['frame'][frame_name+i], self._resize_height, self._resize_width)
					if self.transform is not None:
						batch.append(self.transform(image))

				imgs = torch.unsqueeze(torch.from_numpy(np.concatenate(batch, axis=0)), 0).cuda()

				# if k == 0:
				# 	torch.save(imgs.cpu(), 'imgs_EvaluateDataLoader.pt')
				
				outputs, feas, updated_feas, fea_cat, m_items_test, softmax_score_query, softmax_score_memory, diversity_loss, similarity_loss = self.model.forward(imgs[:,0:3*4], self.m_items_test, None, False)
				# if k == 0:
				# 	torch.save(outputs.cpu(), 'outputs_EvaluateDataLoader.pt')
				mse_imgs = torch.mean(loss_func_mse(outputs, imgs[:,12:])).item()
				mse_feas = similarity_loss.item()
		
				psnr_list[videos_list[video_num].split('/')[-1]].append(psnr(mse_imgs))
				feature_distance_list[videos_list[video_num].split('/')[-1]].append(mse_feas)

		# Measuring the abnormality score and the AUC
		anomaly_score_total_list = list()
		for video in sorted(videos_list):
			video_name = video.split('/')[-1]
			anomaly_score_total_list += score_sum(anomaly_score_list(psnr_list[video_name]), anomaly_score_list_inv(feature_distance_list[video_name]), self.alpha)
		
		# torch.save(torch.from_numpy(np.asarray(anomaly_score_total_list)), 'anomaly_score_total_list_EvaluateDataLoader.pt')
		# exit()

		anomaly_score_total_list = np.asarray(anomaly_score_total_list)
		return anomaly_score_total_list

	def __getitem__(self, index):
		# self.samples store the paths of the frames
		video_name = self.samples[index].split('/')[-2]
		frame_name = int(self.samples[index].split('/')[-1].split('.')[-2])
		
		batch = []
		for i in range(self._time_step+self._num_pred):
			image = np_load_frame(self.videos[video_name]['frame'][frame_name+i], self._resize_height, self._resize_width)
			if self.transform is not None:
				batch.append(self.transform(image))
		if self.is_anomaly_score:
			return np.concatenate(batch, axis=0), self.anomaly_score[index]
		else:
			return np.concatenate(batch, axis=0)
		
		
	def __len__(self):
		return len(self.samples)
