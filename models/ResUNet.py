import torch 
import torch.nn as nn

from .modules import  ResNetBlock
from .attention_modules import  ChannelGate




class ResUNET_channel_attention(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, filters=[32,64,128,256]):
        super(ResUNET_channel_attention, self).__init__()

        ## input and encoder blocks
        self.input_layer = nn.Sequential(
            nn.Conv3d(in_channels, filters[0], kernel_size=3, padding=1),
            nn.InstanceNorm3d(filters[0]),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv3d(in_channels, filters[0], kernel_size=3, padding=1)
        )

        self.residual_conv_1 = ResNetBlock(filters[0], filters[1], stride=2, padding=1)
        self.residual_conv_2 = ResNetBlock(filters[1], filters[2], stride=2, padding=1)


        ## bridge
        self.bridge = ResNetBlock(filters[2], filters[3], stride=2, padding=1)

        ## decoder blocks
        self.upsample_1 = nn.ConvTranspose3d(filters[3],filters[3], kernel_size=2, stride=2)
        self.channel_attention_1 = ChannelGate(filters[3]+filters[2])
        self.up_residual_conv_1 = ResNetBlock(filters[3]+filters[2],filters[2], stride=1, padding=1)
        

        self.upsample_2 = nn.ConvTranspose3d(filters[2],filters[2], kernel_size=2, stride=2)
        self.channel_attention_2 = ChannelGate(filters[2]+filters[1])
        self.up_residual_conv_2 = ResNetBlock(filters[2]+filters[1],filters[1], stride=1, padding=1)
        

        self.upsample_3 = nn.ConvTranspose3d(filters[1],filters[1], kernel_size=2, stride=2)
        self.channel_attention_3 = ChannelGate(filters[1]+filters[0])
        self.up_residual_conv_3 = ResNetBlock(filters[1]+filters[0],filters[0], stride=1, padding=1)
        

        ## output layer
        self.output_layer = nn.Conv3d(filters[0], out_channels, kernel_size=1, stride=1,)
        
        ## output for deep supervision
        self.output_layer_1 = nn.Conv3d(filters[1], out_channels, kernel_size=1, stride=1,)
        self.output_layer_2 = nn.Conv3d(filters[2], out_channels, kernel_size=1, stride=1,)

    def forward(self, x, deep_supervision=False, student=False):
        
        ## Encoder 
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)

        ## Bridge
        x4 = self.bridge(x3)

        ## Decoder
        x4_up = self.upsample_1(x4)
        x5 = torch.cat([x4_up, x3], dim=1)
        x5_attention = self.channel_attention_1(x5)
        x6 = self.up_residual_conv_1(x5_attention)
        


        x6_up = self.upsample_2(x6)
        x7 = torch.cat([x6_up, x2], dim=1)
        x7_attention = self.channel_attention_2(x7)
        x8 = self.up_residual_conv_2(x7_attention)
        


        x8_up = self.upsample_3(x8)
        x9 = torch.cat([x8_up, x1], dim=1)
        x9_attention = self.channel_attention_3(x9)
        x10 = self.up_residual_conv_3(x9_attention)
        

        output = self.output_layer(x10)
        
        if deep_supervision and student:
            output_1 = self.output_layer_1(x8)
            output_2 = self.output_layer_2(x6)
            return [output, output_1, output_2]

        return output if not deep_supervision else [output, x8, x6]




if __name__ == "__main__":
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  image_size = 128
  x = torch.Tensor(2, 3, image_size, image_size, image_size)

  x.to(device)
  print("x size: {}".format(x.size()))
  
  model = ResUNET_channel_attention(in_channels=3, out_channels=4)


  out = model(x)
  print("out size: {}".format(out.size()))