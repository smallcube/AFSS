#timm version=0.6.13
import torch
import torch.nn as nn
import torch.nn.functional as F

import timm.models.vision_transformer
#from timm.models._manipulate import checkpoint_seq
from timm.models.vision_transformer import Block
import numpy as np

from models.CosNormClassifier import CosNorm_Classifier

from timm.models.layers import Mlp

from typing import Optional, Type, OrderedDict

class AUX_Layer(nn.Module):
    def __init__(self, global_pool='token',
                 embed_dim=768,
            num_heads=12,
            mlp_ratio=4.,
            depth=1,
            qkv_bias=True,
            qk_norm=False,
            init_values: Optional[float] = None,
            num_prefix_tokens: bool = True,
            fc_norm=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            proj_drop_rate=0.,
            dpr=None,
            norm_layer=None,
            act_layer=None,
            block_fn=Block, 
            mlp_layer: Type[nn.Module] = Mlp,
            normalized=True,
            num_classes=1000,
            scale=30):
        
        super().__init__()

        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        #norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        #act_layer = act_layer or nn.GELU
        #norm_layer = get_norm_layer(norm_layer) or partial(nn.LayerNorm, eps=1e-6)
        #act_layer = get_act_layer(act_layer) or nn.GELU
        
        self.num_classes = num_classes
        self.global_pool = global_pool
        #self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_prefix_tokens = num_prefix_tokens
        #self.no_embed_class = no_embed_class
        #self.grad_checkpointing = False

        if global_pool == 'map':
            self.attn_pool = AttentionPoolLatent(
                embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
            )
        else:
            self.attn_pool = None

        self.blocks = nn.Sequential(*[
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                init_values=init_values,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_layer=mlp_layer,
            )
            for i in range(depth)])
        
        
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        if normalized:
            self.head = CosNorm_Classifier(embed_dim, num_classes, scale=scale)
            #self.FC = NormedLinear(channel, num_classes, scale=scale)
        else:
            self.head = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        #feature
        x = self.blocks(x)
        x = self.norm(x)
            
        #head

        if self.attn_pool is not None:
            x = self.attn_pool(x)
        elif self.global_pool == 'avg':
            x = x[:, self.num_prefix_tokens:].mean(dim=1)
        elif self.global_pool:
            x = x[:, 0]  # class token
        
        x = self.fc_norm(x)
        x = self.head_drop(x)
        feat = x
        x = self.head(feat)
        return x, feat

def load_model(aux_model, model, aux_depth, selected_layers):
    checkpoint_dict = OrderedDict()
    for idx, layer_num in enumerate(selected_layers):
        key_checkpoint = str(idx)+".blocks."
        for idx2 in range(aux_depth):
            this_selected_layer = layer_num + 1 + idx2

            for key in model.state_dict():
                s = key.split(".")
                if s[0] != "blocks":
                    #print(key)
                    continue
                if int(s[1])==this_selected_layer:
                    this_key_checkpoint = key_checkpoint + str(idx2)
                    for idx3, s3 in enumerate(s):
                        if idx3>=2:
                            this_key_checkpoint += "."+s3
                    checkpoint_dict[this_key_checkpoint] = model.state_dict()[key]
                    #print(this_key_checkpoint)
    x = aux_model.state_dict()
    x.update(checkpoint_dict)
    aux_model.load_state_dict(x, strict=False)
    return aux_model




class VisionTransformer(nn.Module):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, model_name, pretrained=True, selected_layers=[6, 8], aux_depth=2, num_classes=1000, normalized=False, scale=30, **kwargs):
        super().__init__()
        #model_args = get_vit_args(model_name)
        #self.backbone  = _create_vision_transformer(model_name, pretrained=pretrained, **dict(model_args, **kwargs))
        #print('pretrained=', pretrained, "  selected_layers=", selected_layers)
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

        '''
        state_dict = self.backbone.state_dict()
        for key in state_dict:
            print(key)
        '''
        
        #print(self.backbone)
        #self.log_scale = nn.Parameter(torch.ones([]) * np.log(1/0.07))   #borrowed from CLIP, used for cosine similarity

        #print(self.backbone)
        self.selected_layers = selected_layers

        dpr = [x.item() for x in torch.linspace(0, self.backbone.drop_path_rate, self.backbone.depth)]  # stochastic depth decay rule
        #print("dpr=", dpr)
        self.aux_layers = []
        for index in range(len(selected_layers)):
            dprs = [dpr[selected_layers[index]+i+1] for i in range(aux_depth)]
            #dprs += [dpr[-1]]
            
            aux_l = AUX_Layer(global_pool=self.backbone.global_pool,
                                    embed_dim=self.backbone.embed_dim,
                                    num_heads=self.backbone.num_heads,
                                    mlp_ratio=self.backbone.mlp_ratio,
                                    depth=aux_depth,
                                    qkv_bias=self.backbone.qkv_bias,
                                    qk_norm=self.backbone.qk_norm,
                                    init_values = self.backbone.init_values,
                                    num_prefix_tokens=self.backbone.num_prefix_tokens,
                                    fc_norm=self.backbone.fc_norm,
                                    drop_rate=self.backbone.drop_rate,
                                    attn_drop_rate=self.backbone.attn_drop_rate,
                                    proj_drop_rate=self.backbone.proj_drop_rate,
                                    dpr=dprs,
                                    norm_layer=self.backbone.norm_layer,
                                    act_layer=self.backbone.act_layer,
                                    block_fn=Block, 
                                    mlp_layer = self.backbone.mlp_layer,
                                    normalized=normalized,
                                    num_classes=self.backbone.num_classes,
                                    scale=scale)
            
            self.aux_layers += [aux_l]
        
        self.aux_layers = nn.ModuleList([aux_layer for aux_layer in self.aux_layers])
        #print(self.aux_layers[0].state_dict()['blocks.0.norm1.weight'])

        if pretrained:
            self.aux_layers = load_model(self.aux_layers, self.backbone, aux_depth=aux_depth, selected_layers=selected_layers)
        #print(self.aux_layers[0].state_dict()['blocks.0.norm1.weight'])
        
        '''
        state_dict2 = self.aux_layers.state_dict()
        for key in state_dict2:
            print(key)
        '''
        #for idx, aux_layer in enumerate(self.aux_layers):
        #    print(aux_layer)

    def forward(self, x):
        #features
        x = self.backbone.patch_embed(x)
        x = self.backbone._pos_embed(x)
        x = self.backbone.patch_drop(x)
        x = self.backbone.norm_pre(x)
        if self.backbone.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.backbone.blocks, x)
        else:
            logits, features, final_features = [], [], []
            for idx, blk in enumerate(self.backbone.blocks):
                x = blk(x)
                if idx in self.selected_layers:
                    features += [x]
            for idx, aux_layer in enumerate(self.aux_layers):
                logit, feature = aux_layer(features[idx])
                logits += [logit]
                final_features += [feature]

        x = self.backbone.norm(x)

        #head
        if self.backbone.attn_pool is not None:
            x = self.backbone.attn_pool(x)
        elif self.backbone.global_pool == 'avg':
            x = x[:, self.backbone.num_prefix_tokens:].mean(dim=1)
        elif self.backbone.global_pool:
            x = x[:, 0]  # class token
        
        
        x = self.backbone.fc_norm(x)
        x = self.backbone.head_drop(x)
        
        feat = x

        x = self.backbone.head(feat)
        final_features += [feat]
        logits += [x]
        return logits, final_features



def create_model(model_name, pretrained=False, num_classes=100, selected_layers=[5, 7, 9], aux_depth=2,
                normalized=False, scale=30):
    
    print("pretrained=", pretrained)
    model = VisionTransformer(model_name, pretrained, num_classes=num_classes, selected_layers=selected_layers,
                              aux_depth=aux_depth, normalized=normalized, scale=scale)
    return model