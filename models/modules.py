import torch.nn as nn

class DoubleConv(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            #first convolution
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
           
            nn.ReLU(inplace=True),
            
            #2nd convolution
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            )    
        
    def forward(self, x):
        return self.conv(x)

class DoubleConv_GN(nn.Module):
    def __init__(self, in_channels, out_channels, first_stride):
        super(DoubleConv_GN, self).__init__()
        num_of_groups = 6 if out_channels % 6 == 0  else  4 
        self.conv = nn.Sequential(
            #first convolution
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=first_stride, padding=1, bias=False),
            nn.GroupNorm(num_groups=num_of_groups, num_channels=out_channels), ## group normalization
            nn.LeakyReLU(inplace=True),
            
            #2nd convolution
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(num_groups=num_of_groups, num_channels=out_channels), ## group normalization
            nn.LeakyReLU(inplace=True),
            )
        
    def forward(self, x):
        return self.conv(x)


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, padding=1):
        super(ResNetBlock, self).__init__()
        
        num_of_groups_1 = 6 if in_channels % 6 == 0  else  4
        num_of_groups_2 = 6 if out_channels % 6 == 0  else  4 

        self.conv_block = nn.Sequential(
            nn.GroupNorm(num_groups=num_of_groups_1, num_channels=in_channels), ## group normalization
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding),

            nn.GroupNorm(num_groups=num_of_groups_2, num_channels=out_channels), ## group normalization
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )

        self.conv_skip = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding),
            nn.GroupNorm(num_groups=num_of_groups_2, num_channels=out_channels), ## group normalization,
        )
    
    def forward(self, x):
        return self.conv_block(x) + self.conv_skip(x)