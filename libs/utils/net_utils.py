import torch
import torch.nn as nn
import torch.nn.init as init

def weights_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    # print(m)
    if isinstance(m, nn.Conv1d):
        init.normal(m.weight.data)
        init.constant(m.bias.data, 0)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal(m.weight.data)
        init.constant(m.bias.data, 0)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal(m.weight.data)
        init.constant(m.bias.data, 0)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal(m.weight.data)
        init.constant(m.bias.data, 0)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal(m.weight.data)
        init.constant(m.bias.data, 0)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal(m.weight.data)
        init.constant(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal(m.weight.data, mean=1, std=0.02)
        init.constant(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal(m.weight.data, mean=1, std=0.02)
        init.constant(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal(m.weight.data, mean=1, std=0.02)
        init.constant(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal(m.weight.data)
        init.constant(m.bias.data, 0)
    # else:
    #     print('Warning, an unknowned instance!!')
    #     print(m)

class L2Norm(nn.Module):
    def __init__(self,n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        #x /= norm
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out

class TCB(nn.Module):
    """
    Transfer Connection Block Architecture
    This block
    """
    
    def __init__(self, lateral_channels, channles,
                 internal_channels=256):
        """
        :param lateral_channels: number of forward feature channles
        :param channles: number of pyramid feature channles
        :param internal_channels: number of internal channels
        """
        super(TCB, self).__init__()
        # conv + bn + relu
        self.conv1 = nn.Conv2d(lateral_channels, internal_channels,
                               kernel_size=3, padding=1)
        # self.bn1 = nn.BatchNorm2d(internal_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        # ((conv2 + bn2) element-wise add  (deconv + deconv_bn)) + relu
        # batch normalization before element-wise addition
        self.conv2 = nn.Conv2d(internal_channels, internal_channels,
                               kernel_size=3, padding=1)
        # self.bn2 = nn.BatchNorm2d(internal_channels)
        self.deconv = nn.ConvTranspose2d(channles, internal_channels,
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1)
        # self.deconv_bn = nn.BatchNorm2d(internal_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
        # conv + bn + relu
        self.conv3 = nn.Conv2d(internal_channels, internal_channels,
                               kernel_size=3, padding=1)
        # self.bn3 = nn.BatchNorm2d(internal_channels)
        self.relu3 = nn.ReLU(inplace=True)
        
        # attribution
        self.out_channels = internal_channels
    
    def forward(self, lateral, x):
        # no batchnorm
        lateral_out = self.relu1(self.conv1(lateral))
        # element-wise addation
        out = self.relu2(
            self.conv2(lateral_out) +
            self.deconv(x)
        )
        
        out = self.relu3(self.conv3(out))
        
        return out
