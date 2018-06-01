import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

'''
Implementation Notes:
-Setting track_running_stats to True in BatchNorm layers seems to hurt validation
    performance, perhaps because it does not see enough examples during training
'''


class FactorizedReduction(nn.Module):
    '''
    Reduce both spatial dimensions (width and height) by a factor of 2, and 
    potentially to change the number of output filters

    https://github.com/melodyguan/enas/blob/master/src/cifar10/general_child.py#L129
    '''

    def __init__(self, in_planes, out_planes, stride=2):
        super(FactorizedReduction, self).__init__()

        assert out_planes % 2 == 0, (
        "Need even number of filters when using this factorized reduction.")

        self.in_planes = in_planes
        self.out_planes = out_planes
        self.stride = stride

        if stride == 1:
            self.fr = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_planes, affine=False, track_running_stats=False))
        else:
            self.path1 = nn.Sequential(
                nn.AvgPool2d(1, stride=stride),
                nn.Conv2d(in_planes, out_planes // 2, kernel_size=1, bias=False))

            self.path2 = nn.Sequential(
                nn.AvgPool2d(1, stride=stride),
                nn.Conv2d(in_planes, out_planes // 2, kernel_size=1, bias=False))
            self.bn = nn.BatchNorm2d(out_planes, affine=False, track_running_stats=False)

    def forward(self, x):
        if self.stride == 1:
            return self.fr(x)
        else:
            path1 = self.path1(x)

            # pad the right and the bottom, then crop to include those pixels
            path2 = F.pad(x, pad=(0, 1, 0, 1), mode='constant', value=0.)
            path2 = path2[:, :, 1:, 1:]
            path2 = self.path2(path2)

            out = torch.cat([path1, path2], dim=1)
            out = self.bn(out)
            return out


class ENASLayer(nn.Module):
    '''
    https://github.com/melodyguan/enas/blob/master/src/cifar10/general_child.py#L245
    '''
    def __init__(self, layer_id, in_planes, out_planes):
        super(ENASLayer, self).__init__()

        self.layer_id = layer_id
        self.in_planes = in_planes
        self.out_planes = out_planes

        self.branch_0 = ConvBranch(in_planes, out_planes, kernel_size=3)
        self.branch_1 = ConvBranch(in_planes, out_planes, kernel_size=3, separable=True)
        self.branch_2 = ConvBranch(in_planes, out_planes, kernel_size=5)
        self.branch_3 = ConvBranch(in_planes, out_planes, kernel_size=5, separable=True)
        self.branch_4 = PoolBranch(in_planes, out_planes, 'avg')
        self.branch_5 = PoolBranch(in_planes, out_planes, 'max')

        self.bn = nn.BatchNorm2d(out_planes, affine=False, track_running_stats=False)

    def forward(self, x, prev_layers, sample_arc):
        # sample_arc is a list, where the first value is for the layer type
        #   and the remaining values indicate whether skip connects occur

        layer_type = sample_arc[0]
        if self.layer_id > 0:
            skip_indices = sample_arc[1]
        else:
            skip_indices = []

        if layer_type == 0:
            out = self.branch_0(x)
        elif layer_type == 1:
            out = self.branch_1(x)
        elif layer_type == 2:
            out = self.branch_2(x)
        elif layer_type == 3:
            out = self.branch_3(x)
        elif layer_type == 4:
            out = self.branch_4(x)
        elif layer_type == 5:
            out = self.branch_5(x)
        else:
            raise ValueError("Unknown layer_type {}".format(layer_type))

        for i, skip in enumerate(skip_indices):
            if skip == 1:
                out += prev_layers[i]

        out = self.bn(out)
        return out


class ConvBranch(nn.Module):
    '''
    https://github.com/melodyguan/enas/blob/master/src/cifar10/general_child.py#L483
    '''
    def __init__(self, in_planes, out_planes, kernel_size, separable=False):
        super(ConvBranch, self).__init__()
        assert kernel_size in [3, 5], "Kernel size must be either 3 or 5"

        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.separable = separable

        self.padding = (kernel_size - 1) // 2
        self.groups = in_planes if separable else 1

        self.inp_conv1 = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_planes, affine=False, track_running_stats=False),
            nn.ReLU())

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                      padding=self.padding, groups=self.groups, bias=False),
            nn.BatchNorm2d(out_planes, affine=False, track_running_stats=False),
            nn.ReLU())

    def forward(self, x):
        out = self.inp_conv1(x)
        out = self.out_conv(out)
        return out


class PoolBranch(nn.Module):
    '''
    https://github.com/melodyguan/enas/blob/master/src/cifar10/general_child.py#L546
    '''
    def __init__(self, in_planes, out_planes, avg_or_max):
        super(PoolBranch, self).__init__()

        self.in_planes = in_planes
        self.out_planes = out_planes
        self.avg_or_max = avg_or_max

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_planes, affine=False, track_running_stats=False),
            nn.ReLU())

        if avg_or_max == 'avg':
            self.pool = torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        elif avg_or_max == 'max':
            self.pool = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        else:
            raise ValueError("Unknown pool {}".format(avg_or_max))

    def forward(self, x):
        out = self.conv1(x)
        out = self.pool(out)
        return out


class SharedCNN(nn.Module):
    def __init__(self,
                 num_layers=12,        # how deep the network is
                 num_branches=6,       # number of options per layer
                 out_filters=24,       # number of output filters in each convolutional layer
                 keep_prob=1.0         # dropout probability
                 ):

        super(SharedCNN, self).__init__()

        self.num_layers = num_layers
        self.num_branches = num_branches
        self.out_filters = out_filters
        self.keep_prob = keep_prob

        pool_distance = self.num_layers // 3
        self.pool_layers = [pool_distance - 1, 2 * pool_distance - 1]

        self.stem_conv = nn.Sequential(
            nn.Conv2d(3, out_filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_filters, affine=False, track_running_stats=False))

        self.layers = nn.ModuleList([])
        self.pooled_layers = nn.ModuleList([])

        for layer_id in range(self.num_layers):
            layer = ENASLayer(layer_id, self.out_filters, self.out_filters)
            self.layers.append(layer)

            if layer_id in self.pool_layers:
                for i in range(len(self.layers)):
                    self.pooled_layers.append(FactorizedReduction(self.out_filters, self.out_filters))
                '''
                # Use this for fixed architecture
                    self.pooled_layers.append(FactorizedReduction(self.out_filters, self.out_filters * 2))
                self.out_filters *= 2
                '''

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=1. - self.keep_prob)
        self.classify = nn.Linear(self.out_filters, 10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x, sample_arc):

        x = self.stem_conv(x)

        prev_layers = []
        pool_count = 0
        for layer_id in range(self.num_layers):
            x = self.layers[layer_id](x, prev_layers, sample_arc[str(layer_id)])
            prev_layers.append(x)
            if layer_id in self.pool_layers:
                for i, prev_layer in enumerate(prev_layers):
                    # Go through the outputs of all previous layers and downsample them
                    prev_layers[i] = self.pooled_layers[pool_count](prev_layer)
                    pool_count += 1
                x = prev_layers[-1]

        x = self.global_avg_pool(x)
        x = x.view(x.shape[0], -1)
        x = self.dropout(x)
        out = self.classify(x)

        return out
