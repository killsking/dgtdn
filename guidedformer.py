from guided_filter_pytorch.guided_filter import FastGuidedFilter

from dehazeformer import *
from torch.nn import functional as F
from model_utils import AdaptiveInstanceNorm
from ops import unpixel_shuffle

import torch.nn as nn
from residual_dense_block import SRDB

class BasicBlock(nn.Module):
	def __init__(self, network_depth, dim, depth, num_heads, mlp_ratio=4.,
				 norm_layer=nn.LayerNorm, window_size=8,
				 attn_ratio=0., attn_loc='last', conv_type=None):

		super().__init__()
		self.dim = dim
		self.depth = depth
		self.gf=FastGuidedFilter(r=1)
		self.downsample = nn.Upsample(
            scale_factor=0.5, mode="bilinear", align_corners=True
        )
    
		depth_rate=24
		kernel_size=3
		in_channels=3
		self.conv_out = nn.Conv2d(depth_rate*2, depth_rate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
		self.relu1=nn.ReLU(inplace=True)
		self.relu2=nn.ReLU(inplace=True)
		self.norm1=AdaptiveInstanceNorm(depth_rate)
		self.norm2=AdaptiveInstanceNorm(depth_rate) 
		attn_depth = attn_ratio * depth
		#print(attn_depth,attn_ratio,depth)
		if attn_loc == 'last':
			use_attns = [i >= depth-attn_depth for i in range(depth)]
		elif attn_loc == 'first':
			use_attns = [i < attn_depth for i in range(depth)]
		elif attn_loc == 'middle':
			use_attns = [i >= (depth-attn_depth)//2 and i < (depth+attn_depth)//2 for i in range(depth)]

		# build blocks
		self.blocks = nn.ModuleList([
			TransformerBlock(network_depth=network_depth,
							 dim=dim, 
							 num_heads=num_heads,
							 mlp_ratio=mlp_ratio,
							 norm_layer=norm_layer,
							 window_size=window_size,
							 shift_size=0 if (i % 2 == 0) else window_size // 2,
							 use_attn=use_attns[i], conv_type=conv_type)
			for i in range(depth)])

	def forward(self, x_hr):
		x_lr = self.downsample(x_hr)
   
		x_lr_new=self.norm1(x_lr)
		x_lr_new=self.relu1( x_lr_new)
		for blc in self.blocks:
   
		    x_lr_new = blc(x_lr_new)
    
		g_hr= self.gf(x_lr, x_lr_new, x_hr)
		gx_cat=torch.cat([g_hr,x_hr],1)
		g_hr=self.conv_out(gx_cat)
		g_hr=self.norm2(g_hr)
		g_hr=self.relu2( g_hr)
		x=g_hr+x_hr
   
		return g_hr
   


class DeepGuidedFilterFormer(nn.Module):
    def __init__(self,  radius=1):
        super().__init__()

        
        norm = AdaptiveInstanceNorm
        depth_rate=24
        kernel_size=3
        in_channels=3
        self.conv_in = nn.Conv2d(in_channels, depth_rate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.conv_out = nn.Conv2d(depth_rate, in_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.relu1=nn.ReLU(inplace=True)
        self.block_num=3
        network_depth=50
        dim=depth_rate
        mlp_ratio=2.0
        norm_layer=RLN
        window_size=16
        conv_type='Conv'
        depth=4
        num_heads=4
        attn_ratio=1/4
        
        
        self.blocks = nn.ModuleList([
                   BasicBlock(network_depth=network_depth, dim=dim, depth=depth,
					   			 num_heads=num_heads, mlp_ratio=mlp_ratio,
					   			 norm_layer=norm_layer, window_size=window_size,
					   			 attn_ratio=attn_ratio, attn_loc='last', conv_type=conv_type)
			             for i in range(self.block_num)])

        

    def forward(self, x_hr):
        x_hr=self.conv_in(x_hr)
        #x_hr=self.relu1(x_hr)
        #pixelshuffle_ratio=2
        # Unpixelshuffle
        #x_lr_unpixelshuffled = unpixel_shuffle(x_lr, pixelshuffle_ratio)
        
        for blc in self.blocks:
            x_hr=blc(x_hr)
            
        x_hr=self.conv_out(x_hr)
        # Pixelshuffle
        #y_lr = F.pixel_shuffle(
           # self.lr(x_lr_unpixelshuffled), pixelshuffle_ratio
        #)

        return x_hr
           
   
   
class ConvGuidedFilter(nn.Module):
    """
    Adapted from https://github.com/wuhuikai/DeepGuidedFilter
    """
    def __init__(self, radius=1, norm=nn.BatchNorm2d, conv_a_kernel_size: int = 1):
        super(ConvGuidedFilter, self).__init__()

        self.box_filter = nn.Conv2d(
            3, 3, kernel_size=3, padding=radius, dilation=radius, bias=False, groups=3
        )
        self.conv_a = nn.Sequential(
            nn.Conv2d(
                6,
                32,
                kernel_size=conv_a_kernel_size,
                padding=conv_a_kernel_size // 2,
                bias=False,
            ),
            norm(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                32,
                32,
                kernel_size=conv_a_kernel_size,
                padding=conv_a_kernel_size // 2,
                bias=False,
            ),
            norm(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                32,
                3,
                kernel_size=conv_a_kernel_size,
                padding=conv_a_kernel_size // 2,
                bias=False,
            ),
        )
        self.box_filter.weight.data[...] = 1.0

    def forward(self, x_lr, y_lr, x_hr):
        _, _, h_lrx, w_lrx = x_lr.size()
        _, _, h_hrx, w_hrx = x_hr.size()

        N = self.box_filter(x_lr.data.new().resize_((1, 3, h_lrx, w_lrx)).fill_(1.0))
        ## mean_x
        mean_x = self.box_filter(x_lr) / N
        ## mean_y
        mean_y = self.box_filter(y_lr) / N
        ## cov_xy
        cov_xy = self.box_filter(x_lr * y_lr) / N - mean_x * mean_y
        ## var_x
        var_x = self.box_filter(x_lr * x_lr) / N - mean_x * mean_x

        ## A
        A = self.conv_a(torch.cat([cov_xy, var_x], dim=1))
        ## b
        b = mean_y - A * mean_x

        ## mean_A; mean_b
        mean_A = F.interpolate(A, (h_hrx, w_hrx), mode="bilinear", align_corners=True)
        mean_b = F.interpolate(b, (h_hrx, w_hrx), mode="bilinear", align_corners=True)

        return mean_A * x_hr + mean_b
        


        

class DeepGuideddetail(nn.Module):
    def __init__(self,  radius=1):
        super().__init__()

        
        norm = AdaptiveInstanceNorm

        
        

        #self.lr = dehazeformer_m()
        kernel_size=3
        depth_rate=16
        in_channels=3
        num_dense_layer=4
        growth_rate=16
        growth_rate=16
        
        self.conv_in = nn.Conv2d(in_channels, depth_rate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.conv_out = nn.Conv2d(depth_rate, in_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.rdb1 = SRDB(depth_rate, num_dense_layer, growth_rate)
        self.rdb2 = SRDB(depth_rate, num_dense_layer, growth_rate)
        self.rdb3 = SRDB(depth_rate, num_dense_layer, growth_rate)
        self.rdb4 = SRDB(depth_rate, num_dense_layer, growth_rate)

        self.gf = ConvGuidedFilter(radius, norm=norm)

        self.downsample = nn.Upsample(
            scale_factor=0.5, mode="bilinear", align_corners=True
        )

    def forward(self, x_hr):
        x_lr = self.downsample(x_hr)
        y_lr=self.conv_in(x_lr)
        y_lr=self.rdb1(y_lr)
        y_lr=self.rdb2(y_lr)
        y_lr=self.rdb3(y_lr)
        y_lr=self.rdb4(y_lr)
        y_lr=self.conv_out(y_lr)
        
        #pixelshuffle_ratio=2
        # Unpixelshuffle
        #x_lr_unpixelshuffled = unpixel_shuffle(x_lr, pixelshuffle_ratio)
        
        #y_lr=self.lr(x_lr)
        # Pixelshuffle
        #y_lr = F.pixel_shuffle(
           # self.lr(x_lr_unpixelshuffled), pixelshuffle_ratio
        #)

        return F.tanh( self.gf(x_lr, y_lr, x_hr))
                
        

class DeepGuidedall(nn.Module):
    def __init__(self,  radius=1):
        super().__init__()

        
        norm = AdaptiveInstanceNorm

        
        

        #self.lr = dehazeformer_m()
        kernel_size=3
        depth_rate=16
        in_channels=3
        num_dense_layer=4
        growth_rate=16
        growth_rate=16
        
        self.conv_in = nn.Conv2d(in_channels, depth_rate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.conv_out = nn.Conv2d(depth_rate, in_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.rdb1 = SRDB(depth_rate, num_dense_layer, growth_rate)
        self.rdb2 = SRDB(depth_rate, num_dense_layer, growth_rate)
        self.rdb3 = SRDB(depth_rate, num_dense_layer, growth_rate)
        self.rdb4 = SRDB(depth_rate, num_dense_layer, growth_rate)

        self.gf = ConvGuidedFilter(radius, norm=norm)
        self.lr = dehazeformer_m()

        self.downsample = nn.Upsample(
            scale_factor=0.5, mode="bilinear", align_corners=True
        )
        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )

    def forward(self, x_hr):
        x_lr = self.downsample(x_hr)
        y_lr=self.conv_in(x_lr)
        y_lr=self.rdb1(y_lr)
        y_lr=self.rdb2(y_lr)
        y_lr=self.rdb3(y_lr)
        y_lr=self.rdb4(y_lr)
        y_detail=self.conv_out(y_lr)
        y_base=self.lr(x_lr)
        y_lr=y_base+y_detail
        y_base=self.upsample(y_base)
        
        #pixelshuffle_ratio=2
        # Unpixelshuffle
        #x_lr_unpixelshuffled = unpixel_shuffle(x_lr, pixelshuffle_ratio)
        
        #y_lr=self.lr(x_lr)
        # Pixelshuffle
        #y_lr = F.pixel_shuffle(
           # self.lr(x_lr_unpixelshuffled), pixelshuffle_ratio
        #)

        return F.tanh( self.gf(x_lr, y_lr, x_hr)), y_base   




class DeepGuidednew(nn.Module):
    def __init__(self,  radius=1):
        super().__init__()

        
        norm = AdaptiveInstanceNorm

        
        

        #self.lr = dehazeformer_m()
        kernel_size=3
        depth_rate=16
        in_channels=3
        num_dense_layer=4
        growth_rate=16
        growth_rate=16
        
        self.conv_in = nn.Conv2d(in_channels, depth_rate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.conv_out = nn.Conv2d(depth_rate, in_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.rdb1 = SRDB(depth_rate, num_dense_layer, growth_rate)
        self.rdb2 = SRDB(depth_rate, num_dense_layer, growth_rate)
        self.rdb3 = SRDB(depth_rate, num_dense_layer, growth_rate)
        self.rdb4 = SRDB(depth_rate, num_dense_layer, growth_rate)

        self.gf = ConvGuidedFilter(radius, norm=norm)
        self.lr = dehazeformer_m()

        self.downsample = nn.Upsample(
            scale_factor=0.5, mode="bilinear", align_corners=True
        )
        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )

    def forward(self, x_hr):
        x_lr = self.downsample(x_hr)
        y_lr=self.conv_in(x_lr)
        y_lr=self.rdb1(y_lr)
        y_lr=self.rdb2(y_lr)
        y_lr=self.rdb3(y_lr)
        y_lr=self.rdb4(y_lr)
        y_detail=self.conv_out(y_lr)
        y_base=self.lr(x_lr)
        y_lr=y_base+y_detail
        y_base=self.upsample(y_base)
        
        #pixelshuffle_ratio=2
        # Unpixelshuffle
        #x_lr_unpixelshuffled = unpixel_shuffle(x_lr, pixelshuffle_ratio)
        
        #y_lr=self.lr(x_lr)
        # Pixelshuffle
        #y_lr = F.pixel_shuffle(
           # self.lr(x_lr_unpixelshuffled), pixelshuffle_ratio
        #)

        return  self.gf(x_lr, y_lr, x_hr), y_base               
        
        

class DeepAtrousGuidedFilter(nn.Module):
    def __init__(self,  radius=1):
        super().__init__()

        
        norm = AdaptiveInstanceNorm

        
        

        self.lr = dehazeformer_m()

        self.gf = ConvGuidedFilter(radius, norm=norm)

        self.downsample = nn.Upsample(
            scale_factor=0.5, mode="bilinear", align_corners=True
        )

    def forward(self, x_hr):
        x_lr = self.downsample(x_hr)
        #pixelshuffle_ratio=2
        # Unpixelshuffle
        #x_lr_unpixelshuffled = unpixel_shuffle(x_lr, pixelshuffle_ratio)
        y_lr=self.lr(x_lr)
        # Pixelshuffle
        #y_lr = F.pixel_shuffle(
           # self.lr(x_lr_unpixelshuffled), pixelshuffle_ratio
        #)

        return F.tanh( self.gf(x_lr, y_lr, x_hr))