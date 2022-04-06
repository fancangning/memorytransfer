from typing import Optional, Any, Tuple
import numpy as np
import os
import sys
import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
from model import utils
from .memory_final_spatial_sumonly_weight_ranking_top1 import *

class GradientReverseFunction(Function):

	@staticmethod
	def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
		ctx.coeff = coeff
		output = input * 1.0
		return output

	@staticmethod
	def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
		return grad_output.neg() * ctx.coeff, None

class GradientReverseLayer(nn.Module):
	def __init__(self):
		super(GradientReverseLayer, self).__init__()

	def forward(self, *input):
		return GradientReverseFunction.apply(*input)

class WarmStartGradientReverseLayer(nn.Module):
	def __init__(self, alpha: Optional[float] = 1.0, lo: Optional[float] = 0.0, hi: Optional[float] = 1., max_iters: Optional[int] = 1000., auto_step: Optional[bool] = False):
		super(WarmStartGradientReverseLayer, self).__init__()
		self.alpha = alpha
		self.lo = lo
		self.hi = hi
		self.iter_num = 0
		self.max_iters = max_iters
		self.auto_step = auto_step

	def forward(self, input: torch.Tensor) -> torch.Tensor:
		coeff = np.float(2.0*(self.hi-self.lo)/(1.0+np.exp(-self.alpha*self.iter_num/self.max_iters))-(self.hi-self.lo)+self.lo)
		if self.auto_step:
			self.step()
		return GradientReverseFunction.apply(input, coeff)

	def step(self):
		"""Increase iteration number :math:'i' by 1"""
		self.iter_num += 1

class DomainDiscriminator(nn.Module):
	"""
	Distinguish whether the input features come from the source domain or the target domain.
	The source domain label is 0 and the target domain label is 1.

	Parameters:
		- **in_feature**(int): dimension of the input features
		- **hidden_size**(int): dimension of the hidden features

	Shape:
		- Inputs:(minibatch, 'in_feature')
		- Outputs:(minibatch, 1)
	"""
	def __init__(self, in_feature: int, hidden_size: int):
		super(DomainDiscriminator, self).__init__()
		self.layer1 = nn.Linear(in_feature, hidden_size)
		self.bn1 = nn.BatchNorm1d(hidden_size)
		self.relu1 = nn.ReLU()
		self.layer2 = nn.Linear(hidden_size, hidden_size)
		self.bn2 = nn.BatchNorm1d(hidden_size)
		self.relu2 = nn.ReLU()
		self.layer3 = nn.Linear(hidden_size, 1)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.relu1(self.bn1(self.layer1(x)))
		x = self.relu2(self.bn2(self.layer2(x)))
		y = self.sigmoid(self.layer3(x))
		return y

	def get_parameters(self) -> list:
		return [{'params': self.parameters(), 'lr_mult': 1.}]


class Encoder(torch.nn.Module):
	def __init__(self, t_length = 5, n_channel =3):
		super(Encoder, self).__init__()
		
		def Basic(intInput, intOutput):
			return torch.nn.Sequential(
				torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
				torch.nn.BatchNorm2d(intOutput),
				torch.nn.ReLU(inplace=False),
				torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
				torch.nn.BatchNorm2d(intOutput),
				torch.nn.ReLU(inplace=False)
			)
		
		def Basic_(intInput, intOutput):
			return torch.nn.Sequential(
				torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
				torch.nn.BatchNorm2d(intOutput),
				torch.nn.ReLU(inplace=False),
				torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
			)
		
		self.moduleConv1 = Basic(n_channel*(t_length-1), 64)
		self.modulePool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

		self.moduleConv2 = Basic(64, 128)
		self.modulePool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
		
		self.moduleConv3 = Basic(128, 256)
		self.modulePool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

		self.moduleConv4 = Basic_(256, 512)
		self.moduleBatchNorm = torch.nn.BatchNorm2d(512)
		self.moduleReLU = torch.nn.ReLU(inplace=False)
		
	def forward(self, x):
		
		# tensorConv1 batch_sizex64x256x256
		tensorConv1 = self.moduleConv1(x)
		tensorPool1 = self.modulePool1(tensorConv1)
		
		# tensorConv2 batch_sizex128x128x128
		tensorConv2 = self.moduleConv2(tensorPool1)
		tensorPool2 = self.modulePool2(tensorConv2)
		
		# tensorConv3 batch_sizex256x64x64
		tensorConv3 = self.moduleConv3(tensorPool2)
		tensorPool3 = self.modulePool3(tensorConv3)
		
		# tensorConv4 batch_sizex512x32x32
		tensorConv4 = self.moduleConv4(tensorPool3)
		
		return tensorConv4, tensorConv1, tensorConv2, tensorConv3

	
	
class Decoder(torch.nn.Module):
	def __init__(self, t_length = 5, n_channel =3):
		super(Decoder, self).__init__()
		
		def Basic(intInput, intOutput):
			return torch.nn.Sequential(
				torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
				torch.nn.BatchNorm2d(intOutput),
				torch.nn.ReLU(inplace=False),
				torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
				torch.nn.BatchNorm2d(intOutput),
				torch.nn.ReLU(inplace=False)
			)
				
		
		def Gen(intInput, intOutput, nc):
			return torch.nn.Sequential(
				torch.nn.Conv2d(in_channels=intInput, out_channels=nc, kernel_size=3, stride=1, padding=1),
				torch.nn.BatchNorm2d(nc),
				torch.nn.ReLU(inplace=False),
				torch.nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=1),
				torch.nn.BatchNorm2d(nc),
				torch.nn.ReLU(inplace=False),
				torch.nn.Conv2d(in_channels=nc, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
				torch.nn.Tanh()
			)
		
		def Upsample(nc, intOutput):
			return torch.nn.Sequential(
				torch.nn.ConvTranspose2d(in_channels = nc, out_channels=intOutput, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
				torch.nn.BatchNorm2d(intOutput),
				torch.nn.ReLU(inplace=False)
			)
	  
		self.moduleConv = Basic(1024, 512)
		self.moduleUpsample4 = Upsample(512, 256)

		self.moduleDeconv3 = Basic(512, 256)
		self.moduleUpsample3 = Upsample(256, 128)

		self.moduleDeconv2 = Basic(256, 128)
		self.moduleUpsample2 = Upsample(128, 64)

		self.moduleDeconv1 = Gen(128,n_channel,64)
		
		
		
	def forward(self, x, skip1, skip2, skip3):
		
		tensorConv = self.moduleConv(x)

		tensorUpsample4 = self.moduleUpsample4(tensorConv)
		cat4 = torch.cat((skip3, tensorUpsample4), dim = 1)
		
		tensorDeconv3 = self.moduleDeconv3(cat4)
		tensorUpsample3 = self.moduleUpsample3(tensorDeconv3)
		cat3 = torch.cat((skip2, tensorUpsample3), dim = 1)
		
		tensorDeconv2 = self.moduleDeconv2(cat3)
		tensorUpsample2 = self.moduleUpsample2(tensorDeconv2)
		cat2 = torch.cat((skip1, tensorUpsample2), dim = 1)
		
		output = self.moduleDeconv1(cat2)

				
		return output
	


class convAE(torch.nn.Module):
	def __init__(self, n_channel =3,  t_length = 5, memory_size = 10, feature_dim = 512, key_dim = 512, temp_update = 0.1, temp_gather=0.1):
		super(convAE, self).__init__()

		self.encoder = Encoder(t_length, n_channel)
		self.decoder = Decoder(t_length, n_channel)
		self.memory = Memory(memory_size,feature_dim, key_dim, temp_update, temp_gather)
	   

	def forward(self, x, keys, discriminator=None, train=True):
		# fea, skip1, skip2, skip3 8x512x32x32 8x64x256x256 8x128x128x128 8x256x64x64

		fea, skip1, skip2, skip3 = self.encoder(x)
		moduleUniversalPool = torch.nn.AdaptiveMaxPool2d((fea.size()[-2], fea.size()[-1]))
		skip1_cat = moduleUniversalPool(skip1)
		skip2_cat = moduleUniversalPool(skip2)
		skip3_cat = moduleUniversalPool(skip3)
		fea_cat = torch.cat((skip1_cat, skip1_cat, skip2_cat, skip3_cat, fea), 1)
		
		# fea_cat 8x1024x32x32
		if discriminator:
			domain_output = discriminator(fea_cat.clone().detach().permute(0, 2, 3, 1).reshape(-1, fea_cat.size()[1]))
			# domain_output 8*32*32x1
			entropy = utils.entropy(domain_output).clone().detach()
		else:
			entropy = torch.ones(fea.size(0)*fea.size(-2)*fea.size(-1), 1).cuda()

		if train:
			updated_fea, keys, softmax_score_query, softmax_score_memory, diversity_loss, similarity_loss = self.memory(fea, keys, entropy, train)
			output = self.decoder(updated_fea, skip1, skip2, skip3)
			
			return output, fea, updated_fea, fea_cat, keys, softmax_score_query, softmax_score_memory, diversity_loss, similarity_loss
		
		#test
		else:
			updated_fea, keys, softmax_score_query, softmax_score_memory, diversity_loss, similarity_loss = self.memory(fea, keys, entropy, train)
			output = self.decoder(updated_fea, skip1, skip2, skip3)
			
			return output, fea, updated_fea, fea_cat, keys, softmax_score_query, softmax_score_memory, diversity_loss, similarity_loss
