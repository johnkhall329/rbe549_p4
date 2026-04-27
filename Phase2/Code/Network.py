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
            input_size=1037,#49152,#24576, 
            hidden_size=1024,#64, 
            num_layers=2,
            batch_first=True)
        
        self.rnnIMU = nn.LSTM(
            input_size=6, 
            hidden_size=6,
            num_layers=2,
            batch_first=True)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=7, stride=2, padding=3), # 480x260
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2), # 240x130
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 120x65
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 2)) # Forces the output to 4x2 regardless of input size
        )
        
        self.linear1 = nn.Linear(1024, 128)
        self.linear2 = nn.Linear(128, 6)
        #self.linear3 = nn.Linear(128, 6)
        #self.linear3.cuda()
        
        self.flow_model = raft_small(weights=Raft_Small_Weights.DEFAULT)

        for param in self.flow_model.parameters(): # freeze RAFT params
            param.requires_grad = False

        self.hidden_state = None
        

    def forward(self, image, imu, xyzQ):
        # image (Batch, 2, 3, H, W)
        # imu (Batch, 10, 6)
        # xyzQ (1,1,7)

        batch_size, _, C, H, W = image.size()
        
        ## Input1: Feed image pairs to RAFT
        img1 = image[:,0,:,:,:]
        img2 = image[:,1,:,:,:]
        flow_list = self.flow_model(img1, img2)
        flow_out = flow_list[-1]
        # c_out = flow_out.view(batch_size,-1)
        c_out = self.feature_extractor(flow_out)
        c_out = c_out.view(batch_size,-1)
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
        
        r_out, (h_n, h_c) = self.rnn(cat_out, self.hidden_state)
        self.hidden_state = (h_n.detach(), h_c.detach())
        l_out1 = self.linear1(r_out[:,-1,:])
        l_out2 = self.linear2(l_out1)
        #l_out3 = self.linear3(l_out2)

        return l_out2

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = DeepVIO()
    net = net.to(device)

    img = torch.randn((2,2,3,520,960), device=device) # Width and Weight may change
    imu = torch.randn((2,10,6), device=device)
    xyzQ = torch.randn((2,1,7), device=device)

    out = net(img, imu, xyzQ)
    print(out)
    out = net(img, imu, xyzQ)
    print(out)