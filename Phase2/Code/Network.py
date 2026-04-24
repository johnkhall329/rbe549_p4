import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

import os

class DeepVIO(nn.module):
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
        
        
        
        checkpoint = None
        checkpoint_pytorch = '/notebooks/model/FlowNet2-C_checkpoint.pth.tar'
        #checkpoint_pytorch = '/notebooks/data/model/FlowNet2-SD_checkpoint.pth.tar'
        if os.path.isfile(checkpoint_pytorch):
            checkpoint = torch.load(checkpoint_pytorch,\
                                map_location=lambda storage, loc: storage.cuda(0))
            best_err = checkpoint['best_EPE']
        else:
            print('No checkpoint')

        
        self.flownet_c = FlowNetC.FlowNetC(batchNorm=False)
        self.flownet_c.load_state_dict(checkpoint['state_dict'])
        self.flownet_c.cuda()

    def forward(self, image, imu, xyzQ):
        batch_size, timesteps, C, H, W = image.size()
        
        ## Input1: Feed image pairs to FlownetC
        c_in = image.view(batch_size, timesteps * C, H, W)
        c_out = self.flownet_c(c_in)
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

