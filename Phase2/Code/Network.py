import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
import sys
import os

import os

class DeepVIO(nn.Module):
    def __init__(self):
        super(DeepVIO, self).__init__()
        
        self.rnn = nn.LSTM(
            input_size=49165,#49152,#24576, 
            hidden_size=1024,#64, 
            num_layers=2,
            batch_first=True)
        self.rnn.cuda()
        
        self.rnnIMU = nn.LSTM(
            input_size=6, 
            hidden_size=6,
            num_layers=2,
            batch_first=True)
        self.rnnIMU.cuda()
        
        self.linear1 = nn.Linear(1024, 128)
        self.linear2 = nn.Linear(128, 6)
        #self.linear3 = nn.Linear(128, 6)
        self.linear1.cuda()
        self.linear2.cuda()
        #self.linear3.cuda()
        
        self.flow_model = raft_small(weights=Raft_Small_Weights.DEFAULT)
        self.flow_model.cuda()

        for param in self.flow_model.parameters(): # freeze RAFT params
            param.requires_grad = False
        

    def forward(self, image, imu, xyzQ):
        batch_size, timesteps, C, H, W = image.size()
        
        ## Input1: Feed image pairs to RAFT
        img1 = image[:,0,:,:,:]
        img2 = image[:,1,:,:,:]
        flow_list = self.flow_model(img1, img2)
        flow_out = flow_list[-1]
        c_out = flow_out.view(batch_size,-1)
        #print('c_out', c_out.shape)
        
        ## Input2: Feed IMU records to LSTM
        imu_out, (imu_n, imu_c) = self.rnnIMU(imu)
        imu_out = imu_out[:, -1, :]
        #print('imu_out', imu_out.shape)
        imu_out = imu_out.unsqueeze(1)
        #print('imu_out', imu_out.shape)
        
        
        ## Combine the output of input1 and 2 and feed it to LSTM
        #r_in = c_out.view(batch_size, timesteps, -1)
        r_in = c_out.view(batch_size, 1, -1)
        #print('r_in', r_in.shape)
        

        cat_out = torch.cat((r_in, imu_out), 2)#1 1 49158
        cat_out = torch.cat((cat_out, xyzQ), 2)#1 1 49165
        
        r_out, (h_n, h_c) = self.rnn(cat_out)
        l_out1 = self.linear1(r_out[:,-1,:])
        l_out2 = self.linear2(l_out1)
        #l_out3 = self.linear3(l_out2)

        return l_out2

if __name__ == '__main__':
    net = DeepVIO()