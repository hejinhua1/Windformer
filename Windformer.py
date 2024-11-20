'''
Windformer: A Transformer-based Neural Network for Wind Speed and Wind Direction Forecasting
'''
import sys
sys.path.append("..")
import torch.nn.functional as F
from torch import linspace, meshgrid, stack, flatten, sum, abs, concat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch
import os
from torch.utils.data import Dataset, DataLoader
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn, optim
from tqdm import tqdm, trange
from data.LoadERA5 import WindDataset
from tool import Data_normalizer, Weighted_loss
import time
from datetime import datetime

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=(3, 3, 3), stride=(1,1,1), padding=(1,1,1), bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=(3, 3, 3), stride=(1,1,1), padding=(1,1,1), bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
# x = torch.randn(32, 24, 13, 32, 32) # N,C,D,H,W
# model = DoubleConv(24, 24)
# y = model(x)
# print(y.shape)


class Inputlayer(nn.Module):
    def __init__(self, n_channels, out_channels, T=4):
        super().__init__()
        self.down = nn.Sequential(DoubleConv(n_channels, out_channels),
                                  nn.MaxPool3d((2, 2, 2)),
                                  DoubleConv(out_channels, out_channels)
        )
        self.layers = nn.ModuleList()
        for t in range(T):
            layer = self.down
            self.layers.append(layer)


    def forward(self, x):
        B, T, C, D, H, W = x.shape
        y = []
        for i in range(T):
            layer_i = self.layers[i]
            x_i = layer_i(x[:, i, :, :, :, :])
            y.append(x_i)
        y = torch.stack(y, dim=1) # B,T,C,D//2,H//2,W//2
        y = y.permute(0, 1, 3, 4, 5, 2).contiguous() # B,T,D//2,H//2,W//2,C
        return y
# x = torch.randn(32, 4, 6, 13, 32, 32) # N,T,C,D,H,W
# # model = nn.Conv3d(in_channels=24, out_channels=12, kernel_size=(2, 3, 3), stride=(2,1,1), padding=(0,1,1), bias=True)
# model = Inputlayer(6, 96)
# y = model(x) # B,T,D//2,H//2,W//2,C
# print(y.shape)
class Outputlayer(nn.Module):
    def __init__(self, n_channels, out_channels, T=4, D=6, hyper_res=2):
        super().__init__()
        self.layers = nn.ModuleList()
        for t in range(T):
            self.layers.append(nn.ConvTranspose3d(n_channels, n_channels//2, kernel_size=hyper_res, stride=hyper_res))

        self.up = nn.Sequential(DoubleConv(hyper_res*T*D, out_channels),
                                DoubleConv(out_channels, out_channels)
        )
        self.conv = nn.Conv3d(n_channels//2, 2, kernel_size=(1, 1, 1), bias=True)
        self.D = D

    def forward(self, x):
        B, T, D, H, W, n_channels = x.shape # n_channels=2*dim
        assert D == self.D, "D must be equal to the given D"
        x = x.permute(0, 1, 5, 2, 3, 4)  # B,T,n_channels,D,H,W
        y = []
        for i in range(T):
            layer_i = self.layers[i]
            x_i = layer_i(x[:, i, :, :, :, :])
            y.append(x_i)
        y = torch.stack(y, dim=1) # B,T,dim,2*D,2*H,2*W
        y = y.permute(0, 1, 3, 2, 4, 5).contiguous() # B,T,2*D,dim,2*H,2*W
        y = y.view(B, -1, n_channels//2, 2*H, 2*W) # B,T*2*D,dim,2*H,2*W
        y = self.up(y) # B,out_channels,dim,2*H,2*W
        y = y.permute(0, 2, 1, 3, 4) # B,dim,out_channels,2*H,2*W
        y = self.conv(y) # B,2,out_channels,2*H,2*W
        y = y.permute(0, 2, 1, 3, 4) # B,out_channels,2,2*H,2*W
        return y

# x = torch.randn(32, 4, 6, 16, 16, 192) # N,T,D,H,W,C
# # model = nn.Conv3d(in_channels=24, out_channels=12, kernel_size=(2, 3, 3), stride=(2,1,1), padding=(0,1,1), bias=True)
# model = Outputlayer(192, 24)
# y = model(x) # B,T,D//2,H//2,W//2,C
# print(y.shape)
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size=(2, 2, 4, 4)):
    """
    Args:
        x: (B, T, D, H, W, C)
        window_size (tuple): window size

    Returns:
        windows: (num_windows*B, window_size[0], window_size[1], window_size[2], C)
    """
    B, T, D, H, W, C = x.shape
    assert T % window_size[0] == 0, "D must be divisible by window_size[0]"
    assert D % window_size[1] == 0, "D must be divisible by window_size[0]"
    assert H % window_size[2] == 0, "H must be divisible by window_size[1]"
    assert W % window_size[3] == 0, "W must be divisible by window_size[2]"

    x = x.view(B, T // window_size[0], window_size[0],
               D // window_size[1], window_size[1],
               H // window_size[2], window_size[2],
               W // window_size[3], window_size[3], C)
    windows = x.permute(0, 1, 3, 5, 7, 2, 4, 6, 8, 9).contiguous().view(-1, window_size[0], window_size[1], window_size[2], window_size[3], C)
    return windows


def window_reverse(windows, window_size, T, D, H, W):
    """
    Args:
        windows: (num_windows*B, window_size[0], window_size[1], window_size[2], window_size[3], C)
        window_size (tuple): Window size

        T (int): Depth of image
        D (int): Depth of image
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, T, D, H, W, C)
    """
    B = int(windows.shape[0] / (T * D * H * W / window_size[0] / window_size[1] / window_size[2] / window_size[3]))
    x = windows.view(B, T // window_size[0], D // window_size[1], H // window_size[2],
                        W // window_size[3], window_size[0], window_size[1],
                        window_size[2], window_size[3], -1)
    x = x.permute(0, 1, 5, 2, 6, 3, 7, 4, 8, 9).contiguous().view(B, T, D, H, W, -1)
    return x



class WindTransformer(nn.Module):
    def __init__(self, dim=96,
                 depth=(2, 6, 6, 2),
                 num_heads=(6, 12, 12, 6),
                 in_chal=6,
                 out_chal=24,
                 mlp_ratio=(4., 2., 2., 4.),
                 drop=(0.2, 0.2, 0.2, 0.2),
                 attn_drop=(0.1, 0.1, 0.1, 0.1),
                 drop_path_list=(0.1, 0.13, 0.17, 0.2)):
        super().__init__()

        # Input layer, convolute the input image to a 4*6*16*16*96 tensor
        self._input_layer = Inputlayer(in_chal, dim)

        # Four basic layers
        self.layer1 = WindSpecificLayer(dim=dim, input_resolution=(4,6,16,16), depth=depth[0], num_heads=num_heads[0], window_size=(2,2,8,8),
                                         mlp_ratio=mlp_ratio[0], drop=drop[0], attn_drop=attn_drop[0], drop_path=drop_path_list[0])
        self.layer2 = WindSpecificLayer(dim=2*dim, input_resolution=(4,6,8,8), depth=depth[1], num_heads=num_heads[1], window_size=(2,2,4,4),
                                         mlp_ratio=mlp_ratio[1], drop=drop[1], attn_drop=attn_drop[1], drop_path=drop_path_list[1])
        self.layer3 = WindSpecificLayer(dim=2*dim, input_resolution=(4,6,8,8), depth=depth[2], num_heads=num_heads[2], window_size=(2,2,4,4),
                                         mlp_ratio=mlp_ratio[2], drop=drop[2], attn_drop=attn_drop[2], drop_path=drop_path_list[2])
        self.layer4 = WindSpecificLayer(dim=dim, input_resolution=(4,6,16,16), depth=depth[3], num_heads=num_heads[3], window_size=(2,2,8,8),
                                         mlp_ratio=mlp_ratio[3], drop=drop[3], attn_drop=attn_drop[3], drop_path=drop_path_list[3])

        # Upsample and downsample
        self.upsample = UpSample(dim=2*dim)
        self.downsample = DownSample(dim=dim)

        # Patch Recovery
        self._output_layer = Outputlayer(2*dim, out_chal)

    def forward(self, x):
        '''Backbone architecture'''
        B, T, C, D, H, W = x.shape # (B, 4, 6, 13, 32, 32)
        # Convolute the input fields
        x = self._input_layer(x)   # (B, 4, 6, 16, 16, C), C = 96

        # Store the tensor for skip-connection
        skip = x

        # Encoder, composed of two layers
        # Layer 1, shape (B, 4, 6, 16, 16, C), C = 96
        x = self.layer1(x)

        # Downsample from (4, 6, 16, 16) to (4, 6, 8, 8)
        x = self.downsample(x)

        # Layer 2, shape (B, 4, 6, 8, 8, 2*C), C = 96
        x = self.layer2(x)

        # Decoder, composed of two layers
        # Layer 3, shape (B, 4, 6, 8, 8, 2*C), C = 96
        x = self.layer3(x)

        # Upsample from (4, 6, 8, 8) to (4, 6, 16, 16)
        x = self.upsample(x)

        # Layer 4, shape (B, 4, 6, 16, 16, C), C = 96
        x = self.layer4(x)

        # Skip connect, in last dimension(C from 96 to 192)
        x = torch.cat((skip, x), dim=5) # (B, 4, 6, 16, 16, C+C)

        # Recover the output fields
        x = self._output_layer(x) # (B, 24, 2, 32, 32)
        return x



class DownSample(nn.Module):
    def __init__(self, dim=96):
        """
        Initialize the DownSample module.

        Args:
            dim (int): The dimensionality of the input data (default is 96).
        """
        super().__init__()
        # A linear function and a layer normalization
        self.linear = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x):
        """
        Forward pass of the DownSample module.

        Args:
            x (tensor): Input data tensor of shape [B, T, D, H, W, C].

        Returns:
            tensor: Output data tensor of shape [B, T, D, H//2, W//2, 2*C].
        """
        x0 = x[:, :, :, 0::2, 0::2, :]  # B, T, D, H/2, W/2, C
        x1 = x[:, :, :, 1::2, 0::2, :]  # B, T, D, H/2, W/2, C
        x2 = x[:, :, :, 0::2, 1::2, :]  # B, T, D, H/2, W/2, C
        x3 = x[:, :, :, 1::2, 1::2, :]  # B, T, D, H/2, W/2, C
        x = torch.cat([x0, x1, x2, x3], dim=5)  # B, T, D, H/2, W/2, 4*C

        x = self.norm(x)
        x = self.linear(x)  # B, T, D, H//2, W//2, 2*C
        return x

class UpSample(nn.Module):
    def __init__(self, dim=96):
        super().__init__()
        '''Up-sampling operation'''
        # Linear layers without bias to increase channels of the data
        self.linear1 = nn.Linear(dim, dim * 2, bias=False)

        # Normalization
        self.norm = nn.LayerNorm(dim * 2)

    def forward(self, x):
        # Call the linear functions to increase channels of the data
        B, T, D, H, W, C = x.shape
        x = self.linear1(x)
        # Call the layer normalization
        x = self.norm(x)
        x = x.permute(3, 4, 5, 0, 1, 2) # H, W, C, B, T, D
        x = x.view(H * 2, W * 2, C // 2, B, T, D) # H * 2, W * 2, C // 2, B, T, D
        x = x.permute(3, 4, 5, 0, 1, 2) # B, T, D, H * 2, W * 2, C // 2
        return x



class WindSpecificLayer(nn.Module):
    def __init__(self, depth, dim, input_resolution, num_heads, window_size, mlp_ratio=4.0, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        '''Basic layer of our network, contains 2 or 6 blocks'''
        self.depth = depth

        # Construct basic blocks
        self.blocks = nn.ModuleList([
            WindSpecificBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=(0,0,0,0) if (i % 2 == 0) else (window_size[0]//2, window_size[1]//2, window_size[2]//2, window_size[3]//2),
                                 mlp_ratio=mlp_ratio,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path)
            for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class WindSpecificBlock(nn.Module):
    r""" Wind SpecificBlock Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (tuple): Window size.
        shift_size (tuple): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=(2,2,6,12), shift_size=(1,1,3,6),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 fused_window_process=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = WindAttention4D(
            dim, heads=num_heads, attn_drop=attn_drop, proj_drop=drop, window_size=window_size)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size[0] > 0:
            # calculate attention mask for TSW-MSA
            T, D, H, W = self.input_resolution
            img_mask = torch.zeros((1, T, D, H, W, 1))  # 1 T D H W 1
            t_slices = (slice(0, -self.window_size[0]),
                        slice(-self.window_size[0], -self.shift_size[0]),
                        slice(-self.shift_size[0], None))
            d_slices = (slice(0, -self.window_size[1]),
                        slice(-self.window_size[1], -self.shift_size[1]),
                        slice(-self.shift_size[1], None))
            h_slices = (slice(0, -self.window_size[2]),
                        slice(-self.window_size[2], -self.shift_size[2]),
                        slice(-self.shift_size[2], None))
            w_slices = (slice(0, -self.window_size[3]),
                        slice(-self.window_size[3], -self.shift_size[3]),
                        slice(-self.shift_size[3], None))
            cnt = 0
            for t in t_slices:
                for d in d_slices:
                    for h in h_slices:
                        for w in w_slices:
                            img_mask[:, t, d, h, w, :] = cnt
                            cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size[0], window_size[1], window_size[2], window_size[3], 1
            mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2] * self.window_size[3])
            # nW, window_size[0] * window_size[1] * window_size[2] * window_size[3]
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            # nW, window_size[0] * window_size[1] * window_size[2], window_size[0] * window_size[1] * window_size[2]
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
        self.fused_window_process = fused_window_process

    def forward(self, x):
        B, T, D, H, W, C = x.shape
        L = T * D * H * W
        x = x.view(B, L, C)
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, T, D, H, W, C)

        # cyclic shift
        if self.shift_size[0] > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1],
                                              -self.shift_size[2], -self.shift_size[3]), dims=(1, 2, 3, 4))
            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size[0], window_size[1], window_size[2], window_size[3], C
        else:
            shifted_x = x
            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size[0], window_size[1], window_size[2], C

        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2] * self.window_size[3], C)
        # nW*B, window_size[0] * window_size[1] * window_size[2]* window_size[3], C

        # W-MSA/TSW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size[0] * window_size[1] * window_size[2]* window_size[3], C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], self.window_size[2], self.window_size[3], C)

        # reverse cyclic shift
        if self.shift_size[0] > 0:
            shifted_x = window_reverse(attn_windows, self.window_size, T, D, H, W)  # B T' D' H' W' C
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1],
                                              self.shift_size[2], self.shift_size[3]), dims=(1, 2, 3, 4))
        else:
            shifted_x = window_reverse(attn_windows, self.window_size, T, D, H, W)  # B T' D' H' W' C
            x = shifted_x
        x = x.view(B, T * D * H * W, C)
        x = shortcut + self.drop_path(x)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.view(B, T, D, H, W, C)
        return x



class WindAttention4D(nn.Module):
    def __init__(self, dim, heads, attn_drop=0., proj_drop=0., window_size=(2, 2, 6, 6)):
        super().__init__()
        '''
        4D window attention with the Wind-Specific bias,
        see https://github.com/microsoft/Swin-Transformer for the official implementation of 2D window attention.
        '''
        # Initialize several operations
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # Store several attributes
        self.num_heads = heads
        self.dim = dim
        self.scale = (dim // heads) ** -0.5
        self.window_size = window_size

        # input_shape is current shape of the self.forward function
        # You can run your code to record it, modify the code and rerun it
        # Record the number of different window types
    #     self.type_of_windows = (input_shape[0] // window_size[0]) * (input_shape[1] // window_size[1])
    #
    #     # For each type of window, we will construct a set of parameters according to the paper
    #     self.earth_specific_bias = ConstructTensor(shape=(
    #         (2 * window_size[2] - 1) * window_size[1] * window_size[1] * window_size[0] * window_size[0],
    #         self.type_of_windows, heads))
    #
    #     # Making these tensors to be learnable parameters
    #     self.earth_specific_bias = nn.Parameter(self.earth_specific_bias)
    #
    #     # Initialize the tensors using Truncated normal distribution
    #     TruncatedNormalInit(self.earth_specific_bias, std=0.02)
    #
    #     # Construct position index to reuse self.earth_specific_bias
    #     self.position_index = self._construct_index()
    #
    # def _construct_index(self):
    #     ''' This function construct the position index to reuse symmetrical parameters of the position bias'''
    #     # Index in the pressure level of query matrix
    #     coords_zi = RangeTensor(self.window_size[0])
    #     # Index in the pressure level of key matrix
    #     coords_zj = -RangeTensor(self.window_size[0]) * self.window_size[0]
    #
    #     # Index in the latitude of query matrix
    #     coords_hi = RangeTensor(self.window_size[1])
    #     # Index in the latitude of key matrix
    #     coords_hj = -RangeTensor(self.window_size[1]) * self.window_size[1]
    #
    #     # Index in the longitude of the key-value pair
    #     coords_w = RangeTensor(self.window_size[2])
    #
    #     # Change the order of the index to calculate the index in total
    #     coords_1 = stack(meshgrid([coords_zi, coords_hi, coords_w]))
    #     coords_2 = stack(meshgrid([coords_zj, coords_hj, coords_w]))
    #     coords_flatten_1 = flatten(coords_1, start_dimension=1)
    #     coords_flatten_2 = flatten(coords_2, start_dimension=1)
    #     coords = coords_flatten_1[:, :, None] - coords_flatten_2[:, None, :]
    #     coords = TransposeDimensions(coords, (1, 2, 0))
    #
    #     # Shift the index for each dimension to start from 0
    #     coords[:, :, 2] += self.window_size[2] - 1
    #     coords[:, :, 1] *= 2 * self.window_size[2] - 1
    #     coords[:, :, 0] *= (2 * self.window_size[2] - 1) * self.window_size[1] * self.window_size[1]
    #
    #     # Sum up the indexes in three dimensions
    #     self.position_index = sum(coords, dim=-1)
    #
    #     # Flatten the position index to facilitate further indexing
    #     self.position_index = flatten(self.position_index)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attention = (q @ k.transpose(-2, -1))

        # # self.earth_specific_bias is a set of neural network parameters to optimize.
        # EarthSpecificBias = self.earth_specific_bias[self.position_index]
        #
        # # Reshape the learnable bias to the same shape as the attention matrix
        # EarthSpecificBias = torch.reshape(EarthSpecificBias, (
        #     self.window_size[0] * self.window_size[1] * self.window_size[2],
        #     self.window_size[0] * self.window_size[1] * self.window_size[2], self.type_of_windows, self.head_number))
        # EarthSpecificBias = TransposeDimensions(EarthSpecificBias, (2, 3, 0, 1))
        # EarthSpecificBias = reshape(EarthSpecificBias, target_shape=[1] + EarthSpecificBias.shape)
        #
        # # Add the Earth-Specific bias to the attention matrix
        # attention = attention + EarthSpecificBias

        if mask is not None:
            nW = mask.shape[0]
            attention = attention.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attention = attention.view(-1, self.num_heads, N, N)
            attention = self.softmax(attention)
        else:
            attention = self.softmax(attention)

        attention = self.attn_drop(attention)

        x = (attention @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# x1 = torch.randn(32, 4, 6, 16, 16, 96) # (B, 4, 6, 16, 16, C)
# lay1 = WindSpecificLayer(dim=96, input_resolution=(4,6,16,16), depth=2, num_heads=12, window_size=(2,2,8,8))
# output1 = lay1(x1)
# print(output1.shape) #(B, 4, 6, 16, 16, C)
#
#
# x2 = torch.randn(32, 4, 6, 8, 8, 96) # (B, 4, 6, 16, 16, C)
# lay2 = WindSpecificLayer(dim=96, input_resolution=(4,6,8,8), depth=2, num_heads=12, window_size=(2,2,4,4))
# output2 = lay2(x2)
# print(output2.shape) #(B, 4, 6, 8, 8, C)

if __name__ == '__main__':
    # x = torch.randn(32, 4, 6, 13, 32, 32) # B,T,C,D,H,W
    # model = WindTransformer(out_chal=1)
    # y = model(x)
    # print(y.shape)
    last_memory = 0
    Resume = False
    if Resume:
        resume_epoch = 0
        checkpoint_name = 'Windformer_h48_39.pt'
    else:
        resume_epoch = 0
    # 迭代次数和检查点保存间隔
    epoch_size, batch_size = 50, 40
    checkpoint_interval = 1
    M = 4  # given the M time steps before time t
    N = 48  # predicts the N time steps after time t
    checkpoint_prefix = 'Windformer_h{}_'.format(N)
    log_path = "/home/hjh/Tyformer/logs/Windformer_h{}_log".format(N)
    Norm_type = 'std' # 'maxmin' or 'std'
    # 设置检查点路径和文件名前缀
    checkpoint_path = "/home/hjh/Tyformer/checkpoints/"
    # Get the current date and time
    current_datetime = datetime.now()
    # Format the current date and time to display only hours and minutes
    formatted_datetime = current_datetime.strftime('%Y-%m-%d %H')

    # 设置GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 模型定义和训练
    model = WindTransformer(out_chal=24).to(device)
    model = nn.DataParallel(model)  # 使用DataParallel包装模型
    # print('Model:', get_memory_diff())
    opt = optim.Adam(model.parameters(), lr=1e-4)
    if Resume:
        # 加载之前保存的模型参数和优化器状态
        checkpoint_file = os.path.join(checkpoint_path, checkpoint_name)
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        # opt.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        pass
    # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.1, patience=5, verbose=True)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epoch_size, verbose=True)
    criterion = nn.L1Loss()
    weighted_loss = Weighted_loss(loss_weight=(5.5287,4.7130))
    normalizer = Data_normalizer()

    trainset = WindDataset(flag='train', Norm_type=Norm_type, M=M, N=N)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valiset = WindDataset(flag='vali', Norm_type=Norm_type, M=M, N=N)
    valiloader = DataLoader(valiset, batch_size=batch_size, shuffle=False)

    # 训练循环
    f = open(log_path + formatted_datetime + '.txt', 'a+')  # 打开文件
    f.write('Norm_type:' + Norm_type + '\n')
    f.close()
    for epoch in range(resume_epoch+1, epoch_size + 1):
        f = open(log_path + formatted_datetime + '.txt', 'a+')  # 打开文件
        train_l_sum, test_l_sum, n = 0.0, 0.0, 0
        train_ul_sum, train_vl_sum, test_ul_sum, test_vl_sum = 0.0, 0.0, 0.0, 0.0
        model.train()
        loop = tqdm((trainloader), total=len(trainloader))
        for (x, y) in loop: # x,y torch.Size([B, M, 6, 13, 41, 61])
            ########################################################
            x = x[:, :, :, :, :32, :32] # torch.Size([B, M, 6, 13, 32, 32])
            # print('Input:', get_memory_diff())
            y = y[:, N-24:N, 3:5, -1, :32, :32] # torch.Size([B, N, 2, 32, 32])
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x)
            # print('Output and intermediate:', get_memory_diff())
            ########################################################
            u_loss = criterion(y_hat[:, :, 0, :, :], y[:, :, 0, :, :])
            v_loss = criterion(y_hat[:, :, 1, :, :], y[:, :, 1, :, :])
            loss = weighted_loss(u_loss, v_loss)
            opt.zero_grad()
            loss.backward()
            opt.step()
            # y_raw, y_hat_raw = y.detach().cpu().numpy(), y_hat.detach().cpu().numpy()
            # u_raw, u_hat_raw = normalizer.inverse_target(y_raw[:, :24, :, :], y_hat_raw[:, :24, :, :], target='u', Norm_type=Norm_type)
            # v_raw, v_hat_raw = normalizer.inverse_target(y_raw[:, 24:, :, :], y_hat_raw[:, 24:, :, :], target='v', Norm_type=Norm_type)
            # u_loss = criterion(torch.from_numpy(u_raw), torch.from_numpy(u_hat_raw))
            # v_loss = criterion(torch.from_numpy(v_raw), torch.from_numpy(v_hat_raw))
            # loss = weighted_loss(u_loss, v_loss)
            # loss_num = loss.numpy()
            # loop.set_description(f'Train Epoch: [{epoch}/{epoch_size}] loss: [{loss_num}]')
            # train_l_sum += loss_num*x.shape[0]
            # train_ul_sum += u_loss*x.shape[0]
            # train_vl_sum += v_loss*x.shape[0]
            loss_num = loss.detach().cpu().numpy()
            loop.set_description(f'Train Epoch: [{epoch}/{epoch_size}] loss: [{loss_num}]')
            train_l_sum += loss_num*x.shape[0]
            train_ul_sum += u_loss.detach().cpu().numpy()*x.shape[0]
            train_vl_sum += v_loss.detach().cpu().numpy()*x.shape[0]
            n += x.shape[0]
        train_loss = train_l_sum / n
        train_u_loss = train_ul_sum / n
        train_v_loss = train_vl_sum / n


        n = 0
        model.eval()
        with torch.no_grad():
            loop = tqdm((valiloader), total=len(valiloader))
            for (x, y) in loop:
                ########################################################
                x = x[:, :, :, :, :32, :32]  # torch.Size([B, M, 6, 13, 32, 32])
                # print('Input:', get_memory_diff())
                y = y[:, N-24:N, 3:5, -1, :32, :32]  # torch.Size([B, N, 2, 32, 32])
                x = x.to(device)
                y_hat = model(x)
                # print('Output and intermediate:', get_memory_diff())
                ########################################################
                y_raw, y_hat_raw = y.numpy(), y_hat.detach().cpu().numpy()
                u_raw, u_hat_raw = normalizer.inverse_target(y_raw[:, :, 0, :, :], y_hat_raw[:, :, 0, :, :], target='u',
                                                             Norm_type=Norm_type)
                v_raw, v_hat_raw = normalizer.inverse_target(y_raw[:, :, 1, :, :], y_hat_raw[:, :, 1, :, :], target='v',
                                                             Norm_type=Norm_type)
                u_loss = criterion(torch.from_numpy(u_raw), torch.from_numpy(u_hat_raw))
                v_loss = criterion(torch.from_numpy(v_raw), torch.from_numpy(v_hat_raw))
                loss = weighted_loss(u_loss, v_loss)
                loss_num = loss.numpy()
                loop.set_description(f'Test Epoch: [{epoch}/{epoch_size}] loss: [{loss_num}]')
                test_l_sum += loss_num * x.shape[0]
                test_ul_sum += u_loss.detach().cpu().numpy() * x.shape[0]
                test_vl_sum += v_loss.detach().cpu().numpy() * x.shape[0]
                n += x.shape[0]

            f.write('Iter: ' + str(epoch) + ' ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '\n')
            print('Iter:', epoch, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        test_loss = test_l_sum / n
        test_u_loss = test_ul_sum / n
        test_v_loss = test_vl_sum / n
        lr_scheduler.step()
        print("第%d个epoch的学习率：%f" % (epoch, opt.param_groups[0]['lr']))
        if epoch % checkpoint_interval == 0:
            # 保存模型检查点
            checkpoint_name = checkpoint_prefix + str(epoch) + '.pt'
            model_path = os.path.join(checkpoint_path, checkpoint_name)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'train_loss': train_loss,
                'test_loss': test_loss,
            }, model_path)
            print('Checkpoint saved:', model_path)

        f.write('Train loss: ' + str(train_loss) + ' Test loss: ' + str(test_loss) + '\n')
        f.write('Train u loss: ' + str(train_u_loss) + ' Test u loss: ' + str(test_u_loss) + '\n')
        f.write('Train v loss: ' + str(train_v_loss) + ' Test v loss: ' + str(test_v_loss) + '\n')
        print('Train loss:', train_loss, ' Test loss:', test_loss)
        print('===' * 20)
        seg_line = '=======================================================================' + '\n'
        f.write(seg_line)
        f.close()


