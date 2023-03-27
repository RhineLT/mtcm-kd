from torch import nn
from torch.nn import init


__all__ = ['initialize_weights']

def initialize_weights(model):
    """Initialize network weights.
    :param model: network to be initialized
    
    Description: Initialize models weights with xavier initialization
    """
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif isinstance(m, nn.BatchNorm3d):
            init.constant_(m.weight.data, 1.0)
            init.constant_(m.bias.data, 0.0)
        elif isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight.data)
            init.constant_(m.bias.data, 0.0)