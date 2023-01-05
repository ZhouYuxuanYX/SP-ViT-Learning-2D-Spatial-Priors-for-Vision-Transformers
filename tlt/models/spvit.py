from timm.models.registry import register_model
import numpy as np
from .layers import *

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225),
        'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    'SP_ViT_Tiny': _cfg(),
    'SP_ViT': _cfg(),
    'SP_ViT_Medium': _cfg(crop_pct=1.0),
    'SP_ViT_Large': _cfg(crop_pct=1.0),
}

def get_block(block_type, **kargs):
    if block_type=='mha':
        # multi-head attention block
        return MHABlock(**kargs)
    elif block_type=='ffn':
        # feed forward block
        return FFNBlock(**kargs)
    elif block_type=='tr':
        # transformer block
        return Block(**kargs)


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def get_dpr(drop_path_rate,depth,drop_path_decay='linear'):
    if drop_path_decay=='linear':
        # linear dpr decay
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
    elif drop_path_decay=='fix':
        # use fixed dpr
        dpr= [drop_path_rate]*depth
    else:
        # use predefined drop_path_rate list
        assert len(drop_path_rate)==depth
        dpr=drop_path_rate
    return dpr


class SP_ViT(nn.Module):
    """
    Arguements:
        p_emb: different conv based position embedding (default: 4 layer conv)
        skip_lam: residual scalar for skip connection (default: 1.0)
        order: which order of layers will be used (default: None, will override depth if given)
        mix_token: use mix token augmentation for batch of tokens (default: False)
        return_dense: whether to return feature of all tokens with an additional aux_head (default: False)
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., drop_path_decay='linear', hybrid_backbone=None, norm_layer=nn.LayerNorm, p_emb='4_2', head_dim = None,
                 skip_lam = 1.0,order=None, mix_token=False, return_dense=False, local_up_to_layer=0):
        super().__init__()
        self.local_up_to_layer = local_up_to_layer
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.output_dim = embed_dim if num_classes==0 else num_classes
        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            if p_emb=='4_2':
                patch_embed_fn = PatchEmbed4_2
            elif p_emb=='4_2_128':
                patch_embed_fn = PatchEmbed4_2_128
            else:
                patch_embed_fn = PatchEmbedNaive

            self.patch_embed = patch_embed_fn(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1 if self.local_up_to_layer<1 else num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        if order is None:
            dpr=get_dpr(drop_path_rate, depth, drop_path_decay)
            self.blocks = nn.ModuleList([
                Block(
                    dim=embed_dim, num_heads=num_heads, head_dim=head_dim, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, skip_lam=skip_lam, use_psa=True) if i<local_up_to_layer else
                Block(
                    dim=embed_dim, num_heads=num_heads, head_dim=head_dim, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                    skip_lam=skip_lam, use_psa=False)
                for i in range(depth)])
        else:
            # use given order to sequentially generate modules
            dpr=get_dpr(drop_path_rate, len(order), drop_path_decay)
            self.blocks = nn.ModuleList([
                get_block(order[i],
                    dim=embed_dim, num_heads=num_heads, head_dim=head_dim, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, skip_lam=skip_lam)
                for i in range(len(order))])

        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        self.return_dense=return_dense
        self.mix_token=mix_token

        if return_dense:
            self.aux_head=nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if mix_token:
            self.beta = 1.0
            assert return_dense, "always return all features when mixtoken is enabled"

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
    
    def forward_embeddings(self,x):
        x = self.patch_embed(x)
        return x
    def forward_tokens(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        if self.local_up_to_layer<=0:
            x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for u, blk in enumerate(self.blocks):
            if self.local_up_to_layer>0 and u == self.local_up_to_layer :
                x = torch.cat((cls_tokens, x), dim=1)
            x = blk(x)
        x = self.norm(x)
        return x

    def forward_features(self,x):
        # simple forward to obtain feature map (without mixtoken)
        x = self.forward_embeddings(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.forward_tokens(x)
        return x

    def forward(self, x):
        x = self.forward_embeddings(x)

        # token level mixtoken augmentation 
        if self.mix_token and self.training:
            lam = np.random.beta(self.beta, self.beta)
            patch_h, patch_w = x.shape[2],x.shape[3]
            bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
            temp_x = x.clone()
            temp_x[:, :, bbx1:bbx2, bby1:bby2] = x.flip(0)[:, :, bbx1:bbx2, bby1:bby2]
            x = temp_x
        else:
            bbx1, bby1, bbx2, bby2 = 0,0,0,0

        x = x.flatten(2).transpose(1, 2)
        x = self.forward_tokens(x)
        x_cls = self.head(x[:,0])


        if self.return_dense:
            x_aux = self.aux_head(x[:,1:])
            if not self.training:
                return x_cls+0.5*x_aux.max(1)[0]

            # recover the mixed part
            if self.mix_token and self.training:
                x_aux = x_aux.reshape(x_aux.shape[0],patch_h, patch_w,x_aux.shape[-1])
                temp_x = x_aux.clone()
                temp_x[:, bbx1:bbx2, bby1:bby2, :] = x_aux.flip(0)[:, bbx1:bbx2, bby1:bby2, :]
                x_aux = temp_x
                x_aux = x_aux.reshape(x_aux.shape[0],patch_h*patch_w,x_aux.shape[-1])

            return x_cls, x_aux, (bbx1, bby1, bbx2, bby2)
        return x_cls

@register_model
def vit(pretrained=False, **kwargs):
    model = SP_ViT(patch_size=16, embed_dim=384, depth=16, num_heads=6, mlp_ratio=3.,
        p_emb=1, **kwargs)
    model.default_cfg = default_cfgs['SP_ViT']
    return model


@register_model
def lvvit(pretrained=False, **kwargs):
    model = SP_ViT(patch_size=16, embed_dim=384, depth=16, num_heads=6, mlp_ratio=3.,
        p_emb='4_2',skip_lam=2., **kwargs)
    model.default_cfg = default_cfgs['SP_ViT']
    return model

@register_model
def lvvit_s(pretrained=False, **kwargs):
    model = SP_ViT(patch_size=16, embed_dim=384, depth=16, num_heads=6, mlp_ratio=3.,
        p_emb='4_2',skip_lam=2., return_dense=True,mix_token=True, **kwargs)
    model.default_cfg = default_cfgs['SP_ViT']
    return model

@register_model
def lvvit_m(pretrained=False, **kwargs):
    model = SP_ViT(patch_size=16, embed_dim=512, depth=20, num_heads=8, mlp_ratio=3.,
        p_emb='4_2',skip_lam=2., return_dense=True,mix_token=True, **kwargs)
    model.default_cfg = default_cfgs['SP_ViT_Medium']
    return model

@register_model
def lvvit_x(pretrained=False, **kwargs):
    model = SP_ViT(patch_size=16, embed_dim=216, depth=10, num_heads=9, mlp_ratio=4.,
        p_emb='4_2',skip_lam=2., return_dense=True,mix_token=True, **kwargs)
    model.default_cfg = default_cfgs['SP_ViT_Medium']
    return model

@register_model
def lvvit_l(pretrained=False, **kwargs):
    order = ['tr']*24 # this will override depth, can also be set as None
    model = SP_ViT(patch_size=16, embed_dim=768,depth=24, num_heads=12, mlp_ratio=3.,
        p_emb='4_2_128',skip_lam=3., return_dense=True,mix_token=True, order=order, **kwargs)
    model.default_cfg = default_cfgs['SP_ViT_Large']
    return model
