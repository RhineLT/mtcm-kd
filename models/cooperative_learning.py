import torch
import torch.nn as nn
from .attention_modules import ChannelGate

__all__ = ['Cooperative_Learning_Module']

class Cooperative_Learning_Module(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        
        self.channel_attention1 = ChannelGate(in_channels)
        self.channel_attention2 = ChannelGate(in_channels)
        self.channel_attention3 = ChannelGate(in_channels)
        
        self.intermediate_conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1,)
        self.intermediate_conv2 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1,)
        self.intermediate_conv3 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1,)
        
        self.output_layer = nn.Conv3d(out_channels, out_channels, kernel_size=1, stride=1,)
        
        
    def forward(self, x1, x2, x3):
        
        
        x1 = self.channel_attention1(x1) 
        x2 = self.channel_attention2(x2) 
        x3 = self.channel_attention3(x3) 
        
        x1 = self.intermediate_conv1(x1)
        x2 = self.intermediate_conv2(x2)
        x3 = self.intermediate_conv3(x3)
        
        x = self.output_layer(x1+x2+x3)
        
        return x
    
    
    
if __name__ == "__main__":
    x1 = torch.rand(1, 64, 64, 64, 64)
    x2 = torch.rand(1, 64, 64, 64, 64)
    x3 = torch.rand(1, 64, 64, 64, 64)
    
    Cooperative_learning = Cooperative_Learning_Module(64)
    output = Cooperative_learning(x1, x2, x3)
    print(output.shape)
    
    
    