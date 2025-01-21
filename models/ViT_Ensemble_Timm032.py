# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# LiVT: https://github.com/XuZhengzhuo/LiVT
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import timm.models.vision_transformer
from timm.models.vision_transformer import Block

from models.CosNormClassifier import CosNorm_Classifier

class AUX_Layer(nn.Module):
    def __init__(self, global_pool='token',
            embed_dim=768,
            num_heads=12,
            mlp_ratio=4.,
            depth=1,
            qkv_bias=True,
            qk_scale=None,
            fc_norm=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.,
            norm_layer=None,
            act_layer=None,
            block_fn=Block, 
            normalized=True,
            num_classes=1000,
            scale=30):
        super().__init__()

        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        
        self.num_classes = num_classes
        self.global_pool = global_pool
        #self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        #self.num_prefix_tokens = 1 if class_token else 0
        #self.no_embed_class = no_embed_class
        #self.grad_checkpointing = False

        self.block_fn = nn.Sequential(*[
                        block_fn(dim=embed_dim, 
                            num_heads=num_heads, 
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            drop=drop_rate,
                            attn_drop=attn_drop_rate,
                            drop_path=drop_path_rate,
                            norm_layer=norm_layer
                        )
                        for i in range(depth)])
        
        
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        if normalized:
            self.head = CosNorm_Classifier(embed_dim, num_classes, scale=scale)
            #self.FC = NormedLinear(channel, num_classes, scale=scale)
        else:
            self.head = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        #feature
        x = self.block_fn(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            feat = self.fc_norm(x)
        else:
            x = self.norm(x)
            feat = x[:, 0]
        
        #head
        x = self.head(feat)
        return x, feat
        


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, model_name, pretrained=True, global_pool=False, selected_layers=[4, 6, 8], aux_depth=2, num_classes=1000, normalized=False, scale=30, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

        self.log_scale = nn.Parameter(torch.ones([]) * np.log(1/0.07))   #borrowed from CLIP, used for cosine similarity

        #print(self.backbone)
        self.selected_layers = selected_layers
        dpr = [x.item() for x in torch.linspace(0, self.backbone.drop_path_rate, self.backbone.depth)]  # stochastic depth decay rule
        print("len=====", len(dpr))
        
        #global_pool = self.backbone.global_pool
        embed_dim = self.backbone.embed_dim
        num_heads = self.backbone.num_heads
        mlp_ratio = self.backbone.mlp_ratio
        #self.selected_layers = [int(i*depth//classifier_num) for i in range(1, classifier_num)]
        self.log_scale = nn.Parameter(torch.ones([]) * np.log(1/0.07))   #borrowed from CLIP, used for cosine similarity
        self.selected_layers = selected_layers

        self.aux_layer1 = AUX_Layer(global_pool=global_pool,
                                    embed_dim=embed_dim,
                                    num_heads=num_heads,
                                    mlp_ratio=mlp_ratio,
                                    normalized=normalized,
                                    num_classes=num_classes,
                                    drop_path_rate=dpr[self.selected_layers[0]],
                                    depth=aux_depth,
                                    scale=scale)
        
        self.aux_layer2 = AUX_Layer(global_pool=global_pool,
                                    embed_dim=embed_dim,
                                    num_heads=num_heads,
                                    mlp_ratio=mlp_ratio,
                                    normalized=normalized,
                                    num_classes=num_classes,
                                    drop_path_rate=dpr[self.selected_layers[1]],
                                    depth=aux_depth,
                                    scale=scale)
        
        self.norm = self.backbone.norm_layer(embed_dim)
        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = self.backbone.norm_layer
            #embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    
    def calibrated_loss3(self, logits, logits_mixed, features, features_mixed, y_a, y_b=None, 
                        gamma=0.5, lam=1.0, mixed_loss=True, weights=None, base_weight=2.0, mixed_index=None):
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
                        gamma=0.5, lam=1.0, mixed_loss=True, weights=None, base_weight=2.0, mixed_index=None):
        #step 1: feature constrastive learning loss
        batch_size = logits.shape[0]
        logit_scale = self.log_scale.exp()
        features = features / features.norm(dim=1, keepdim=True)
        features_mixed = features_mixed / features_mixed.norm(dim=1, keepdim=True)
        #borrowed from CLIP
        features_logits = logit_scale*features @ features_mixed.t()
        #features_logits = features_logits.detach().clone()
        features_logits = features_logits.view(batch_size, -1)
        
        modulating_factor = torch.softmax(features_logits, dim=-1)
        #features_pt = torch.softmax(features_logits, dim=-1)
        features_ground_truth = torch.arange(batch_size, dtype=torch.long).view(-1, 1).to(logits.device)
        #step 2: supervised learning loss
        modulating_factor = modulating_factor.gather(1, features_ground_truth).detach().clone()

        if mixed_loss:
            logpt = (base_weight+modulating_factor*gamma) * F.log_softmax(logits_mixed, dim=1)
            #logpt = modulating_factor**gamma * F.log_softmax(logits_mixed, dim=1)
            loss = lam*F.nll_loss(logpt, y_a) + (1-lam)*F.nll_loss(logpt, y_b)
        else:
            logpt = (base_weight+modulating_factor*gamma) * F.log_softmax(logits, dim=1)
            #logpt = modulating_factor**gamma * F.log_softmax(logits, dim=1)
            loss = F.nll_loss(logpt, y_a)
        #print('CE_Loss=', loss.item(), "   feature_loss=", features_loss.item())
        return loss, weights
        
    def forward_features(self, x):
        B = x.shape[0]
        x = self.backbone.patch_embed(x)

        cls_tokens = self.backbone.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.backbone.pos_embed
        x = self.backbone.pos_drop(x)

        features = []

        for idx, blk in enumerate(self.backbone.blocks):
            x = blk(x)
            if idx in self.selected_layers:
                features += [x]


        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome, features

    def forward(self, x):
        feat, features = self.forward_features(x)
        logit1, feature1 = self.aux_layer1(features[0])
        logit2, feature2 = self.aux_layer2(features[1])
        logit = self.backbone.head(feat)

        logits = [logit1] + [logit2] + [logit]
        features = [feature1] + [feature2] + [feat]
        
        return logits, features


def create_model(model_name, pretrained=False, global_pool=False, num_classes=100, selected_layers=[5, 7, 9], aux_depth=2,
                normalized=False, scale=30):
    
    print("pretrained=", pretrained)
    model = VisionTransformer(model_name, pretrained, global_pool=global_pool, num_classes=num_classes, selected_layers=selected_layers,
                              aux_depth=aux_depth, normalized=normalized, scale=scale)
    return model
