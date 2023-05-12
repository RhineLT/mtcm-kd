import torch
import torch.nn as nn
from .attention_modules import ChannelGate, CBAM

__all__ = ['Cooperative_Learning_Module']

class Cooperative_Learning_Module(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        
        self.cbam_attention1 = CBAM(in_channels)
        self.cbam_attention2 = CBAM(in_channels)
        self.cbam_attention3 = CBAM(in_channels)
        
        self.intermediate_conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1,)
        self.intermediate_conv2 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1,)
        self.intermediate_conv3 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1,)
        
        self.output_layer = nn.Conv3d(out_channels, out_channels, kernel_size=1, stride=1,)
        
        
    def forward(self, x1, x2, x3):
        
        
        x1 = self.cbam_attention1(x1) 
        x2 = self.cbam_attention2(x2) 
        x3 = self.cbam_attention3(x3) 
        
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
    
    
    