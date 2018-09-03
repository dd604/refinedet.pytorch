import torch
import torch.nn as nn


from vgg import make_vgg_layers, add_extra_layers, \
    key_layer_ids, layers_out_channels
from refinedet import RefineDet as _RefineDet
from libs.utils.net_utils import L2Norm


class VGGRefineDet(_RefineDet):
    """
    """
    def __init__(self, num_classes, phase, cfg):
        _RefineDet.__init__(self, num_classes, phase, cfg)
        
    
    def _init_modules(self, model_path=None, pretrained=True):
        base = nn.ModuleList(make_vgg_layers())
        extra = nn.ModuleList(add_extra_layers())
        # self.base = nn.ModuleList(make_vgg_layers())
        # self.extra = nn.ModuleList(add_extra_layers())
        self.pretrained = pretrained
        self.model_path = model_path
        if self.pretrained == True:
            print("Loading pretrained weights from %s" % (self.model_path))
            state_dict = torch.load(self.model_path)
            base.load_state_dict({k: v for k, v in state_dict.items()
                                       if k in base.state_dict()})
        self.layers_out_channels = layers_out_channels
    
        # construct base network
        assert key_layer_ids[2] == -1 and key_layer_ids[3] == -1, \
            'Must use outputs of the final layers in base and extra.'
        self.layer1 = nn.Sequential(self.base[:key_layer_ids[0]])
        self.layer2 = nn.Sequential(self.base[key_layer_ids[0] : key_layer_ids[1]])
        self.layer3 = nn.Sequential(self.base[key_layer_ids[1]:])
        # sequential, modulelist neseted?
        self.layer4 = nn.Sequential(*extra)
        # L2Norm has been initialized while building.
        self.L2Norm_conv4_3 = L2Norm(512, 8)
        self.L2Norm_conv5_3 = L2Norm(512, 10)
        
        # build pyramid layers and other parts
        _RefineDet._init_part_modules(self)
    
    def _calculate_forward_features(self, x):
        # forward features
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.L2Norm_conv4_3(self.layer3(c2))
        c4 = self.L2Norm_conv5_3(self.layer4(c3))
        forward_features = [c1, c2, c3, c4]
        
        return forward_features