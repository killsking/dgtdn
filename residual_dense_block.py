"""
paper: GridDehazeNet: Attention-Based Multi-Scale Network for Image Dehazing
file: residual_dense_block.py
about: build the Residual Dense Block
author: Xiaohong Liu
date: 01/08/19
"""

# --- Imports --- #
import torch
import torch.nn as nn
import torch.nn.functional as F



    
class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3):
        super().__init__()

        self.conv1 =nn.Conv2d(in_channel, in_channel, kernel_size=kernel_size, padding=(kernel_size - 1) // 2) #ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        return out    
    




        

class WAVSRDB(nn.Module):
    def __init__(self, in_channels, num_dense_layer, growth_rate):
        """

        :param in_channels: input channel size
        :param num_dense_layer: the number of RDB layers
        :param growth_rate: growth_rate
        """
        super(WAVSRDB, self).__init__()
        
        modules = []
        self.haarwav=HaarTransform(in_channels)
        self.haarinwav=InverseHaarTransform(in_channels)
        self.split_channel=in_channels
        kernel_size=3
        dilation=1
        self.conv1 = nn.Conv2d(self.split_channel*1, self.split_channel, kernel_size=kernel_size, padding=dilation, dilation=dilation)
        dilation=2
        self.conv2 = nn.Conv2d(self.split_channel*2, self.split_channel, kernel_size=kernel_size, padding=dilation, dilation=dilation)
        dilation=2
        self.conv3 = nn.Conv2d(self.split_channel*3, self.split_channel, kernel_size=kernel_size,  padding=dilation, dilation=dilation)
        dilation=1
        self.conv4 = nn.Conv2d(self.split_channel*4, self.split_channel, kernel_size=kernel_size, padding=dilation, dilation=dilation)

            
        #self.residual_dense_layers = nn.Sequential(*modules)
        _in_channels=in_channels
        self.calayer=CALayer(in_channels)
        self.palayer=PALayer(in_channels)
        self.conv_1x1 = nn.Conv2d(_in_channels*4, in_channels, kernel_size=1, padding=0)

    def forward(self, x):
        ll,  lh, hl, hh = self.haarwav(x)
        x0=F.relu(self.conv1(ll))
        tmp= torch.cat((lh, x0), 1)
        x1=F.relu(self.conv2(tmp))
        tmp= torch.cat((hl, x0, x1), 1)
        x2=F.relu(self.conv3(tmp))
        tmp= torch.cat((hh, x0, x1, x2), 1)
        x3=F.relu(self.conv4(tmp))
        tmp= torch.cat(( x0, x1, x2, x3), 1)
        
        out = self.haarinwav(tmp)
        out=self.calayer(out)
        out=self.palayer(out)
        out=out+x
        return out
                


        
class CAB(nn.Module):
    def __init__(self, features):
        super(CAB, self).__init__()
        #new_features=features//2
        features=features//2
        self.reduce_fature=nn.Conv2d(features*2, features, kernel_size=1, bias=False)
        self.delta_gen1 = nn.Sequential(
                        nn.Conv2d(features*2, features, kernel_size=1, bias=False),
                        nn.Conv2d(features, 2, kernel_size=3, padding=1, bias=False)
                        )

        self.delta_gen2 = nn.Sequential(
                        nn.Conv2d(features*2, features, kernel_size=1, bias=False),
                        nn.Conv2d(features, 2, kernel_size=3, padding=1, bias=False)
                        )


        #self.delta_gen1.weight.data.zero_()
        #self.delta_gen2.weight.data.zero_()

    # https://github.com/speedinghzl/AlignSeg/issues/7
    # the normlization item is set to [w/s, h/s] rather than [h/s, w/s]
    # the function bilinear_interpolate_torch_gridsample2 is standard implementation, please use bilinear_interpolate_torch_gridsample2 for training.
    def bilinear_interpolate_torch_gridsample(self, input, size, delta=0):
        out_h, out_w = size
        n, c, h, w = input.shape
        s = 1.0
        norm = torch.tensor([[[[w/s, h/s]]]]).type_as(input).to(input.device)
        w_list = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h_list = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h_list.unsqueeze(2), w_list.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + delta.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid)
        return output

    def bilinear_interpolate_torch_gridsample2(self, input, size, delta=0):
        out_h, out_w = size
        n, c, h, w = input.shape
        s = 2.0
        norm = torch.tensor([[[[(out_w-1)/s, (out_h-1)/s]]]]).type_as(input).to(input.device) # not [h/s, w/s]
        w_list = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h_list = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h_list.unsqueeze(2), w_list.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + delta.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid, align_corners=True)
        return output

    def forward(self, low_stage, high_stage):
        h, w = low_stage.size(2), low_stage.size(3)
        high_stage=self.reduce_fature(high_stage)
        high_stage = F.interpolate(input=high_stage, size=(h, w), mode='bilinear', align_corners=True)
        
        concat = torch.cat((low_stage, high_stage), 1)
        delta1 = self.delta_gen1(concat)
        delta2 = self.delta_gen2(concat)
        high_stage = self.bilinear_interpolate_torch_gridsample2(high_stage, (h, w), delta1)
        low_stage = self.bilinear_interpolate_torch_gridsample2(low_stage, (h, w), delta2)

        high_stage += low_stage
        return high_stage

# --- Build dense --- #
class MakeDense(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size=3):
        super(MakeDense, self).__init__()
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=kernel_size, padding=(kernel_size-1)//2)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out

class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
    def forward(self, x):
        y = self.pa(x)
        return x * y

class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y

class SRDBDK(nn.Module):
    def __init__(self, in_channels, num_dense_layer, growth_rate):
        """

        :param in_channels: input channel size
        :param num_dense_layer: the number of RDB layers
        :param growth_rate: growth_rate
        """
        super(SRDBDK, self).__init__()
        
        modules = []
        self.split_channel=in_channels//8
        kernel_size=3
        dilation=1
        self.conv1 = nn.Conv2d(self.split_channel*1, self.split_channel, kernel_size=9, padding=4, dilation=1)
        dilation=2
        self.conv2 = nn.Conv2d(self.split_channel*2, self.split_channel*1, kernel_size=7, padding=3, dilation=1)
        dilation=4
        self.conv3 = nn.Conv2d(self.split_channel*4, self.split_channel*2, kernel_size=5,  padding=2, dilation=1)
        dilation=8
        self.conv4 = nn.Conv2d(self.split_channel*8, self.split_channel*4, kernel_size=3, padding=1, dilation=1)

            
        #self.residual_dense_layers = nn.Sequential(*modules)
        _in_channels=in_channels
        self.calayer=CALayer(in_channels)
        self.palayer=PALayer(in_channels)
        self.conv_1x1 = nn.Conv2d(_in_channels, in_channels, kernel_size=1, padding=0)

    def forward(self, x):
        splited = torch.split(x, [self.split_channel,self.split_channel*1,self.split_channel*2,self.split_channel*4], dim=1)
        x0=F.relu(self.conv1(splited[0]))
        tmp= torch.cat((splited[1], x0), 1)
        x1=F.relu(self.conv2(tmp))
        tmp= torch.cat((splited[2], x0, x1), 1)
        x2=F.relu(self.conv3(tmp))
        tmp= torch.cat((splited[3], x0, x1, x2), 1)
        x3=F.relu(self.conv4(tmp))
        tmp= torch.cat(( x0, x1, x2, x3), 1)
        
        out = self.conv_1x1(tmp)
        out=self.calayer(out)
        out=self.palayer(out)
        out=out+x
        return out
        
                

class SRDB(nn.Module):
    def __init__(self, in_channels, num_dense_layer, growth_rate):
        """

        :param in_channels: input channel size
        :param num_dense_layer: the number of RDB layers
        :param growth_rate: growth_rate
        """
        super(SRDB, self).__init__()
        
        modules = []
        self.split_channel=in_channels//4
        kernel_size=3
        dilation=1
        self.conv1 = nn.Conv2d(self.split_channel*1, self.split_channel, kernel_size=kernel_size, padding=dilation, dilation=dilation)
        dilation=2
        self.conv2 = nn.Conv2d(self.split_channel*2, self.split_channel, kernel_size=kernel_size, padding=dilation, dilation=dilation)
        dilation=4
        self.conv3 = nn.Conv2d(self.split_channel*3, self.split_channel, kernel_size=kernel_size,  padding=dilation, dilation=dilation)
        dilation=8
        self.conv4 = nn.Conv2d(self.split_channel*4, self.split_channel, kernel_size=kernel_size, padding=dilation, dilation=dilation)

            
        #self.residual_dense_layers = nn.Sequential(*modules)
        _in_channels=in_channels
        self.calayer=CALayer(in_channels)
        self.palayer=PALayer(in_channels)
        self.conv_1x1 = nn.Conv2d(_in_channels, in_channels, kernel_size=1, padding=0)

    def forward(self, x):
        splited = torch.split(x, self.split_channel, dim=1)
        x0=F.relu(self.conv1(splited[0]))
        tmp= torch.cat((splited[1], x0), 1)
        x1=F.relu(self.conv2(tmp))
        tmp= torch.cat((splited[2], x0, x1), 1)
        x2=F.relu(self.conv3(tmp))
        tmp= torch.cat((splited[3], x0, x1, x2), 1)
        x3=F.relu(self.conv4(tmp))
        tmp= torch.cat(( x0, x1, x2, x3), 1)
        
        out = self.conv_1x1(tmp)
        out=self.calayer(out)
        out=self.palayer(out)
        #print(out.shape, x.shape)
        out=out+x
        return out


class SRDBN(nn.Module):
    def __init__(self, in_channels, num_dense_layer, growth_rate):
        """

        :param in_channels: input channel size
        :param num_dense_layer: the number of RDB layers
        :param growth_rate: growth_rate
        """
        super(SRDBN, self).__init__()
        
        modules = []
        self.split_channel=in_channels//8
        kernel_size=3
        dilation=1
        self.conv1 = nn.Conv2d(self.split_channel*1, self.split_channel, kernel_size=kernel_size, padding=dilation, dilation=dilation)
        dilation=2
        self.conv2 = nn.Conv2d(self.split_channel*2, self.split_channel, kernel_size=kernel_size, padding=dilation, dilation=dilation)
        dilation=4
        self.conv3 = nn.Conv2d(self.split_channel*3, self.split_channel, kernel_size=kernel_size,  padding=dilation, dilation=dilation)
        dilation=8
        self.conv4 = nn.Conv2d(self.split_channel*4, self.split_channel, kernel_size=kernel_size, padding=dilation, dilation=dilation)
        
        dilation=8
        self.conv5 = nn.Conv2d(self.split_channel*5, self.split_channel, kernel_size=kernel_size, padding=dilation, dilation=dilation)
        
        dilation=4
        self.conv6 = nn.Conv2d(self.split_channel*6, self.split_channel, kernel_size=kernel_size, padding=dilation, dilation=dilation)
        
        dilation=2
        self.conv7 = nn.Conv2d(self.split_channel*7, self.split_channel, kernel_size=kernel_size, padding=dilation, dilation=dilation)
        
        dilation=1
        self.conv8 = nn.Conv2d(self.split_channel*8, self.split_channel, kernel_size=kernel_size, padding=dilation, dilation=dilation)

            
        #self.residual_dense_layers = nn.Sequential(*modules)
        _in_channels=in_channels
        self.calayer=CALayer(in_channels)
        self.palayer=PALayer(in_channels)
        self.conv_1x1 = nn.Conv2d(_in_channels, in_channels, kernel_size=1, padding=0)

    def forward(self, x):
        splited = torch.split(x, self.split_channel, dim=1)
        x0=F.relu(self.conv1(splited[0]))
        tmp= torch.cat((splited[1], x0), 1)
        x1=F.relu(self.conv2(tmp))
        tmp= torch.cat((splited[2], x0, x1), 1)
        x2=F.relu(self.conv3(tmp))
        tmp= torch.cat((splited[3], x0, x1, x2), 1)
        x3=F.relu(self.conv4(tmp))
        tmp= torch.cat(( splited[4],x0, x1, x2, x3), 1)
        x4=F.relu(self.conv5(tmp))
        
        tmp= torch.cat(( splited[5],x0, x1, x2, x3,x4), 1)
        x5=F.relu(self.conv6(tmp))
        
        tmp= torch.cat(( splited[6],x0, x1, x2, x3,x4,x5), 1)
        x6=F.relu(self.conv7(tmp))
        
        tmp= torch.cat(( splited[7],x0, x1, x2, x3,x4,x5,x6), 1)
        x7=F.relu(self.conv8(tmp))
        
       
        tmp= torch.cat(( x0, x1, x2, x3,x4,x5,x6,x7), 1)
        out = self.conv_1x1(tmp)
        out=self.calayer(out)
        out=self.palayer(out)
        out=out+x
        return out


        
                

# --- Build the Residual Dense Block --- #
class RDB(nn.Module):
    def __init__(self, in_channels, num_dense_layer, growth_rate):
        """

        :param in_channels: input channel size
        :param num_dense_layer: the number of RDB layers
        :param growth_rate: growth_rate
        """
        super(RDB, self).__init__()
        _in_channels = in_channels
        modules = []
        for i in range(num_dense_layer):
            modules.append(MakeDense(_in_channels, growth_rate))
            _in_channels += growth_rate
        self.residual_dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(_in_channels, in_channels, kernel_size=1, padding=0)

    def forward(self, x):
        out = self.residual_dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x
        return out
