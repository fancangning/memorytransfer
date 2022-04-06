import torch
import torch.autograd as ag
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import functools
import random
from torch.nn import functional as F

def random_uniform(shape, low, high, cuda):
    x = torch.rand(*shape)
    result_cpu = (high - low) * x + low
    if cuda:
        return result_cpu.cuda()
    else:
        return result_cpu
    
def distance(a, b):
    return torch.sqrt(((a - b) ** 2).sum()).unsqueeze(0)

def distance_batch(a, b):
    bs, _ = a.shape
    result = distance(a[0], b)
    for i in range(bs-1):
        result = torch.cat((result, distance(a[i], b)), 0)
        
    return result

def multiply(x): #to flatten matrix into a vector 
    return functools.reduce(lambda x,y: x*y, x, 1)

def flatten(x):
    """ Flatten matrix into a vector """
    count = multiply(x.size())
    return x.resize_(count)

def index(batch_size, x):
    idx = torch.arange(0, batch_size).long() 
    idx = torch.unsqueeze(idx, -1)
    return torch.cat((idx, x), dim=1)

def MemoryLoss(memory):

    m, d = memory.size()
    memory_t = torch.t(memory)
    similarity = (torch.matmul(memory, memory_t))/2 + 1/2 # 30X30
    identity_mask = torch.eye(m).cuda()
    sim = torch.abs(similarity - identity_mask)
    
    return torch.sum(sim)/(m*(m-1))


class Memory(nn.Module):
    def __init__(self, memory_size, feature_dim, key_dim,  temp_update, temp_gather):
        super(Memory, self).__init__()
        # Constants
        self.memory_size = memory_size
        self.feature_dim = feature_dim
        self.key_dim = key_dim
        self.temp_update = temp_update
        self.temp_gather = temp_gather
        
    def hard_neg_mem(self, mem, i):
        similarity = torch.matmul(mem,torch.t(self.keys_var))
        similarity[:,i] = -1
        _, max_idx = torch.topk(similarity, 1, dim=1)
        
        
        return self.keys_var[max_idx]
    
    def random_pick_memory(self, mem, max_indices):
        
        m, d = mem.size()
        output = []
        for i in range(m):
            flattened_indices = (max_indices==i).nonzero()
            a, _ = flattened_indices.size()
            if a != 0:
                number = np.random.choice(a, 1)
                output.append(flattened_indices[number, 0])
            else:
                output.append(-1)
            
        return torch.tensor(output)
    
    def get_update_query(self, mem, max_indices, update_indices, entropy, score, query, train):
        # max_indices query_size*1
        # update_indices 1*memory_size
        # entropy query_size*1
        max_indices = max_indices.clone().detach()
        entropy = entropy.clone().detach()
        score = score.clone().detach()
        m, d = mem.size()
        if train:
            query_update = torch.zeros((m,d)).cuda()
            # random_update = torch.zeros((m,d)).cuda()
            for i in range(m):
                idx = torch.nonzero(max_indices.squeeze(1)==i)
                a, _ = idx.size()
                if a != 0:
                    query_update[i] = torch.sum(((score[idx,i]/torch.max(score[:,i]))*(entropy[idx,0]/torch.max(entropy[:,0]))*query[idx].squeeze(1)), dim=0)
                else:
                    query_update[i] = 0 
        
       
            return query_update 
    
        else:
            query_update = torch.zeros((m,d)).cuda()
            for i in range(m):
                idx = torch.nonzero(max_indices.squeeze(1)==i)
                a, _ = idx.size()
                if a != 0:
                    query_update[i] = torch.sum(((score[idx,i]/torch.max(score[:,i]))*(entropy[idx,0]/torch.max(entropy[:,0]))*query[idx].squeeze(1)), dim=0)
                else:
                    query_update[i] = 0 
            
            return query_update

    def get_score(self, mem, query):
        bs, h, w, d = query.size() # batch_size*height*weight*channel
        m, d = mem.size() # memory_size*channel
        
        score = torch.matmul(query, torch.t(mem))# b X h X w X m
        score = score.view(bs*h*w, m)# (b X h X w) X m
        
        score_query = F.softmax(score, dim=0) # query_size x memory_size
        score_memory = F.softmax(score, dim=1) # query_size x mamory_size
        
        return score_query, score_memory
    
    def forward(self, query, keys, entropy, train=True):

        batch_size, dims, h, w = query.size() # batch_size X channel X height X width 8 X 512 X 32 X 32
        query = F.normalize(query, dim=1) # notice that the query has been normalized!!!!!!!!!
        query = query.permute(0,2,3,1) # batch_size X height X width X channel 8 X 32 X 32 X 512
        
        #train
        if train:
            #losses
            diversity_loss, similarity_loss = self.memory_loss(query, keys, entropy, train)
            # read
            updated_query, softmax_score_query, softmax_score_memory = self.read(query, keys)
            #update
            updated_memory, diversity_loss_updated_memory = self.update(query, keys, entropy, train)
            
            return updated_query, updated_memory, softmax_score_query, softmax_score_memory, diversity_loss_updated_memory, similarity_loss
        
        #test
        else:
            # loss
            diversity_loss, similarity_loss = self.memory_loss(query, keys, entropy, train)
            
            # read
            updated_query, softmax_score_query,softmax_score_memory = self.read(query, keys)
            
            #update
            updated_memory = keys
            
            return updated_query, updated_memory, softmax_score_query, softmax_score_memory, diversity_loss, similarity_loss
        
        
    
    def update(self, query, keys, entropy, train):
        
        batch_size, h, w, dims = query.size() # batch_size X height X width X channel 
        
        softmax_score_query, softmax_score_memory = self.get_score(keys, query) # query_size x memory_size
        
        query_reshape = query.contiguous().view(batch_size*h*w, dims)
        
        _, gathering_indices = torch.topk(softmax_score_memory, 1, dim=1) # query_size*1
        _, updating_indices = torch.topk(softmax_score_query, 1, dim=0) # 1*memory_size
        
        if train:
             
            query_update = self.get_update_query(keys, gathering_indices, updating_indices, entropy, softmax_score_query, query_reshape, train)
            updated_memory = F.normalize(query_update + keys.clone().detach(), dim=1)
            ones = torch.ones(updated_memory.size(0), updated_memory.size(0), dtype=torch.float32)
            diag = torch.eye(updated_memory.size(0), dtype=torch.float32)
            diversity_loss = torch.sum((torch.matmul(updated_memory, updated_memory.T)*(ones-diag).cuda())**2) / (updated_memory.size(0)*updated_memory.size(0)-updated_memory.size(0))
        
        else:
            query_update = self.get_update_query(keys, gathering_indices, updating_indices, entropy, softmax_score_query, query_reshape, train)
            updated_memory = F.normalize(query_update + keys.clone().detach(), dim=1)
            ones = torch.ones(updated_memory.size(0), updated_memory.size(0), dtype=torch.float32)
            diag = torch.eye(updated_memory.size(0), dtype=torch.float32)
            diversity_loss = torch.sum((torch.matmul(updated_memory, updated_memory.T)*(ones-diag).cuda())**2) / (updated_memory.size(0)*updated_memory.size(0)-10)
        
        return updated_memory.detach(), diversity_loss
        
        
    def pointwise_gather_loss(self, query_reshape, keys, gathering_indices, train):
        n,dims = query_reshape.size() # (b X h X w) X d
        loss_mse = torch.nn.MSELoss(reduction='none')
        
        pointwise_loss = loss_mse(query_reshape, keys[gathering_indices].squeeze(1).detach())
                
        return pointwise_loss
        
    def memory_loss(self, query, keys, entropy, train):
        #####################################################################################################
        # my similarity loss and diversity loss
        #####################################################################################################
        batch_size, h, w, dims = query.size() # batch_size X height X width X channel 8 X 32 X 32 X 512
        cos_similarity = torch.nn.CosineEmbeddingLoss(reduction='none')
        softmax_score_query, softmax_score_memory = self.get_score(keys, query) # query_size x memory_size
    
        query_reshape = query.contiguous().view(batch_size*h*w, dims)
    
        _, argmax_idx = torch.topk(softmax_score_memory, 1, dim=1)
    
        #1st closest memories
        closest_memory_item = keys[argmax_idx[:,0]]

        ones = torch.ones(keys.size(0), keys.size(0), dtype=torch.float32)
        diag = torch.eye(keys.size(0), dtype=torch.float32)
        diversity_loss = torch.sum((torch.matmul(keys, keys.T)*(ones-diag).cuda())**2) / (keys.size(0)*keys.size(0)-keys.size(0))
        similarity_loss = torch.dot(cos_similarity(query_reshape, closest_memory_item, torch.ones(query_reshape.size(0)).cuda()), torch.squeeze(entropy))
        return diversity_loss, similarity_loss

        #####################################################################################################
        # MNAD similarity loss and diversity loss
        #####################################################################################################
        # batch_size, h, w, dims = query.size() # batch_size X height X width X channel 8 X 32 X 32 X 512
        # loss = torch.nn.TripletMarginLoss(margin=1.0)
        # loss_mse = torch.nn.MSELoss()
        # softmax_score_query, softmax_score_memory = self.get_score(keys, query)
    
        # query_reshape = query.contiguous().view(batch_size*h*w, dims)
    
        # _, gathering_indices = torch.topk(softmax_score_memory, 2, dim=1)
    
        # #1st, 2nd closest memories
        # pos = keys[gathering_indices[:,0]]
        # neg = keys[gathering_indices[:,1]]
        # similarity_loss = loss_mse(query_reshape, pos.detach())
        # diversity_loss = loss(query_reshape, pos.detach(), neg.detach())
        
        # return diversity_loss, similarity_loss
        
        
    
    def read(self, query, updated_memory):
        batch_size, h, w, dims = query.size() # b X h X w X d

        softmax_score_query, softmax_score_memory = self.get_score(updated_memory, query) # query_size x memory_size

        query_reshape = query.contiguous().view(batch_size*h*w, dims)
        
        concat_memory = torch.matmul(softmax_score_memory.detach(), updated_memory) # query_size x memory_size   memory_size x dim
        updated_query = torch.cat((query_reshape, concat_memory), dim = 1) # (b X h X w) X 2d
        updated_query = updated_query.view(batch_size, h, w, 2*dims)
        updated_query = updated_query.permute(0,3,1,2)
        
        return updated_query, softmax_score_query, softmax_score_memory
