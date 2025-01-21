from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor

from models.AUX_Layer_DDP import *



def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        #_log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.SyncBatchNorm
            #norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        feat_dim = 512 * block.expansion
        self.aux_layer2 = Aux_Layer1(block.expansion*128, feat_dim, num_classes)
        self.aux_layer3 = Aux_Layer2(block.expansion*256, feat_dim, num_classes)
        #self.log_scale = nn.Parameter(torch.ones([]) * np.log(1/0.07))   #borrowed from CLIP, used for cosine similarity

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)
    
    def calibrated_loss3(self, logits, logits_mixed, features, features_mixed, y_a, y_b=None, 
                        gamma=0.5, lam=1.0, mixed_loss=True, weights=None, base_weight=2.0):
        #step 1: feature constrastive learning loss
        batch_size = logits.shape[0]
        logit_scale = self.log_scale.exp()
        features = features / features.norm(dim=1, keepdim=True)
        features_mixed = features_mixed / features_mixed.norm(dim=1, keepdim=True)
        #borrowed from CLIP
        features_logits = logit_scale*features @ features_mixed.t()
        #features_logits = features_logits.detach().clone()
        features_logits = features_logits.unsqueeze(2)
        
        if weights is None:
            weights = features_logits
            modulating_factor=0.0
        else:
            weights = torch.cat((weights, features_logits), 2)
            modulating_factor = torch.mean(weights, dim=2).view(batch_size, -1)
            modulating_factor = torch.softmax(modulating_factor, dim=-1)
            #features_pt = torch.softmax(features_logits, dim=-1)
            features_ground_truth = torch.arange(batch_size, dtype=torch.long).view(-1, 1).to(logits.device)
            #step 2: supervised learning loss
            modulating_factor = modulating_factor.gather(1, features_ground_truth).detach().clone()

        if mixed_loss:
            logpt = (base_weight+modulating_factor)**gamma * F.log_softmax(logits_mixed, dim=1)
            #logpt = modulating_factor**gamma * F.log_softmax(logits_mixed, dim=1)
            loss = lam*F.nll_loss(logpt, y_a) + (1-lam)*F.nll_loss(logpt, y_b)
        else:
            logpt = (base_weight+modulating_factor)**gamma * F.log_softmax(logits, dim=1)
            #logpt = modulating_factor**gamma * F.log_softmax(logits, dim=1)
            loss = F.nll_loss(logpt, y_a)
        #print('CE_Loss=', loss.item(), "   feature_loss=", features_loss.item())
        return loss, weights

    def calibrated_loss4(self, logits, logits_mixed, features, features_mixed, y_a, y_b=None, 
                        gamma=0.5, lam=1.0, mixed_loss=True, weights=None, base_weight=2.0):
        #step 1: feature constrastive learning loss
        batch_size = logits.shape[0]
        logit_scale = self.log_scale.exp()
        features = features.detach().clone()
        features_mixed = features_mixed.detach().clone()
        features = features / features.norm(dim=1, keepdim=True)
        features_mixed = features_mixed / features_mixed.norm(dim=1, keepdim=True)
        
        #borrowed from CLIP
        features_logits = features @ features_mixed.t()
        #features_logits = features_logits.detach().clone()
        features_logits = features_logits.unsqueeze(2)
        
        '''
        if weights is None:
            weights = features_logits
        else:
            weights = torch.cat((weights, features_logits), 2)
        '''
        weights = features_logits
        modulating_factor = torch.mean(weights, dim=2).view(batch_size, -1)
        modulating_factor = torch.softmax(modulating_factor, dim=-1)
        #features_pt = torch.softmax(features_logits, dim=-1)
        features_ground_truth = torch.arange(batch_size, dtype=torch.long).view(-1, 1).to(logits.device)
        #step 2: supervised learning loss
        modulating_factor = modulating_factor.gather(1, features_ground_truth).detach().clone()

        if mixed_loss:
            logpt = (base_weight+modulating_factor)**gamma * F.log_softmax(logits_mixed, dim=1)
            #logpt = modulating_factor**gamma * F.log_softmax(logits_mixed, dim=1)
            loss = lam*F.nll_loss(logpt, y_a) + (1-lam)*F.nll_loss(logpt, y_b)
        else:
            logpt = (base_weight+modulating_factor)**gamma * F.log_softmax(logits, dim=1)
            #logpt = modulating_factor**gamma * F.log_softmax(logits, dim=1)
            loss = F.nll_loss(logpt, y_a)
        #print('CE_Loss=', loss.item(), "   feature_loss=", features_loss.item())
        return loss, weights
    
    def calibrated_loss5(self, logits, logits_mixed, features, features_mixed, y_a, y_b=None, 
                        gamma=0.5, lam=1.0, mixed_loss=True, weights=None, base_weight=2.0):
        #step 1: feature constrastive learning loss
        batch_size, num_classes = logits.shape[0], logits.shape[1]
        logit_scale = self.log_scale.exp()
        features = features / features.norm(dim=1, keepdim=True)
        features_mixed = features_mixed / features_mixed.norm(dim=1, keepdim=True)
        #borrowed from CLIP
        features_logits = logit_scale*features @ features_mixed.t()
        #features_logits = features_logits.detach().clone()
        features_logits = features_logits.unsqueeze(2)
        
        '''
        if weights is None:
            weights = features_logits
        else:
            weights = torch.cat((weights, features_logits), 2)
        '''
        weights = features_logits
        #modulating_factor = torch.mean(weights, dim=2).view(batch_size, -1)
        modulating_factor = torch.softmax(modulating_factor, dim=-1)
        #features_pt = torch.softmax(features_logits, dim=-1)
        features_ground_truth = torch.arange(batch_size, dtype=torch.long).view(-1, 1).to(logits.device)
        #step 2: supervised learning loss
        modulating_factor = modulating_factor.gather(1, features_ground_truth).detach().clone()

        target1_one_hot = F.one_hot(y_a, num_classes=num_classes).view(batch_size, -1).float()

        if mixed_loss:
            pt = (base_weight+modulating_factor)**gamma * torch.sigmoid(logits_mixed)
            target2_one_hot = F.one_hot(y_b, num_classes=num_classes).view(batch_size, -1).float()
            loss1 = target1_one_hot *torch.log(pt)+(1-target1_one_hot)*torch.log(1-pt)
            loss1 = torch.sum(loss1, dim=-1).mean()
            loss2 = target2_one_hot *torch.log(pt)+(1-target2_one_hot)*torch.log(1-pt)
            loss2 = torch.sum(loss2, dim=-1).mean()

            #logpt = modulating_factor**gamma * F.log_softmax(logits_mixed, dim=1)
            loss = lam*loss1 + (1-lam)*loss2
        else:
            pt = (base_weight+modulating_factor)**gamma * torch.sigmoid(logits)
            #logpt = modulating_factor**gamma * F.log_softmax(logits, dim=1)
            loss1 = target1_one_hot *torch.log(pt)+(1-target1_one_hot)*torch.log(1-pt)
            loss = torch.sum(loss1, dim=-1).mean()
        #print('CE_Loss=', loss.item(), "   feature_loss=", features_loss.item())
        return loss, weights
    
   

    def forward(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        logits2, features2 = self.aux_layer2(x)

        x = self.layer3(x)
        logits3, features3 = self.aux_layer3(x)

        x = self.layer4(x)

        x = self.avgpool(x)

        feature = torch.flatten(x, 1)
        logit = self.fc(feature)

        logits = [logits2] + [logits3] + [logit]
        features = [features2] + [features3] + [feature]

        return logits, features


def create_model(depth, num_classes=1000, groups=1, width_per_group=64):
    print('Loading Scratch ResNet_Ensemble/ResNeXt_Ensemble Model.')
    if depth==18:
        resnet = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, groups=groups, width_per_group=width_per_group)
    elif depth==50:
        resnet = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, groups=groups, width_per_group=width_per_group)
    elif depth==101:
        resnet = ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, groups=groups, width_per_group=width_per_group)

    return resnet

