import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

DROPOUT_FLOPS = 4
LAYER_NORM_FLOPS = 5
ACTIVATION_FLOPS = 8
SOFTMAX_FLOPS = 5

class Mlp(nn.Module):
    '''
    MLP with support to use group linear operator
    '''

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., group=1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        if group == 1:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)
        else:
            self.fc1 = GroupLinear(in_features, hidden_features, group)
            self.fc2 = GroupLinear(hidden_features, out_features, group)
        self.act = act_layer()

        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# Our proposed SP_SA
class SP_SA(nn.Module):
    def __init__(self, dim, num_heads=8, predict_center=False, qkv_bias=False, qk_scale=None, attn_drop=0.,
                 proj_drop=0.,
                 locality_strength=1., use_local_init=True, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.w1 = nn.Linear(2, dim, bias=True)
        self.w2 = nn.Parameter(torch.zeros(dim, 1))
        self.b2 = nn.Parameter(torch.ones(num_heads))

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.locality_strength = locality_strength
        self.apply(self._init_weights)
        self.act = nn.ReLU()
        self.predict_center = predict_center

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, padding_mask=None):
        B, N, C = x.shape
        if not hasattr(self, 'rel_indices') or self.rel_indices.size(1) != N:
            self.get_rel_indices(N)

        attn = self.get_attention(x)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def get_attention(self, x):
        B, N, C = x.shape
        qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        pos_emb = self.w1(self.rel_indices).view(1, N, N, self.num_heads, -1)

        pos_map = torch.einsum('nm,hijnm->hijn', (self.w2.view(self.num_heads, -1), self.act(pos_emb))) + self.b2

        pos_map = pos_map.repeat(B, 1, 1, 1)

        q, k = qk[0], qk[1]
        patch_score = (q @ k.transpose(-2, -1)) * self.scale
        patch_score_masked = (patch_score * pos_map.permute(0, 3, 1, 2)).softmax(dim=-1)

        attn = self.attn_drop(patch_score_masked)

        return attn

    def get_attention_map(self, x, return_map=False, return_distances=False, return_single=False, index=0):
        B, N, C = x.shape
        if not hasattr(self, 'rel_indices') or self.rel_indices.size(1) != N:
            self.get_rel_indices(N)
        if return_single:
            attn_map = self.get_attention(x)[index]
        else:
            attn_map = self.get_attention(x).mean(0)  # average over batch

        distances = (self.rel_indices[0, :, :, 0] ** 2 + self.rel_indices[0, :, :, 1] ** 2) ** 0.5
        # the i,j of the rel_indices matrix means i th patch to j th patch (14x14 = 196 after splitting the image to patches)
        dist = torch.einsum('nm,hnm->h', (distances, attn_map))
        dist /= distances.size(0)
        if return_map:
            return dist, attn_map
        elif return_distances:
            return dist, distances
        else:
            return dist

    def get_rel_indices(self, num_patches):
        img_size = int(num_patches ** .5)
        rel_indices = torch.zeros(1, num_patches, num_patches, 2)
        ind = torch.arange(img_size).view(1, -1) - torch.arange(img_size).view(-1, 1)
        indx = ind.repeat(img_size, img_size)
        indy = ind.repeat_interleave(img_size, dim=0).repeat_interleave(img_size, dim=1)
        rel_indices[:, :, :, 1] = indy.unsqueeze(0)
        rel_indices[:, :, :, 0] = indx.unsqueeze(0)
        device = self.qk.weight.device
        # top-down direction is defined as positive in y, left-right is defined as positive in x
        self.rel_indices = rel_indices.to(device)


class Attention(nn.Module):
    '''
    Multi-head self-attention
    from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    with some modification to support different num_heads and head_dim.
    '''

    def __init__(self, dim, num_heads=8, head_dim=None, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        if head_dim is not None:
            self.head_dim = head_dim
        else:
            head_dim = dim // num_heads
            self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, self.head_dim * self.num_heads * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.head_dim * self.num_heads, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, padding_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        # B,heads,N,C/heads
        q, k, v = qkv[0], qkv[1], qkv[2]

        # trick here to make q@k.t more stable
        attn = ((q * self.scale) @ k.transpose(-2, -1))
        if padding_mask is not None:
            attn = attn.view(B, self.num_heads, N, N)
            attn = attn.masked_fill(
                padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )
            attn_float = attn.softmax(dim=-1, dtype=torch.float32)
            attn = attn_float.type_as(attn)
        else:
            attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.head_dim * self.num_heads)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# From Conditional Positional Encoding: https://github.com/Meituan-AutoML/CPVT
class PosCNN(nn.Module):
    def __init__(self, in_chans, embed_dim=768, s=1):
        super(PosCNN, self).__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim, 3, s, 1, bias=True, groups=embed_dim), )
        self.s = s

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1)

        if N % 2 == 1:
            feat_token = x[:, :, 1:]
        else:
            feat_token = x
        cnn_feat = feat_token.view(B, C, H, W)
        if self.s == 1:
            if N % 2 == 1:
                x[:, :, 1:] = (self.proj(cnn_feat) + cnn_feat).view(B, C, H * W)
            else:
                x = (self.proj(cnn_feat) + cnn_feat).view(B, C, H * W)
        else:
            if N % 2 == 1:
                x[:, :, 1:] = self.proj(cnn_feat).view(B, C, H * W)
            else:
                x = self.proj(cnn_feat).view(B, C, H * W)
        x = x.transpose(1, 2)
        return x

    def no_weight_decay(self):
        return ['proj.%d.weight' % i for i in range(4)]


class Block(nn.Module):
    '''
    Pre-layernorm transformer block
    '''

    def __init__(self, dim, num_heads, head_dim=None, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, group=1, skip_lam=1., use_psa=False):
        super().__init__()
        self.dim = dim
        self.mlp_hidden_dim = int(dim * mlp_ratio)
        self.skip_lam = skip_lam

        self.norm1 = norm_layer(dim)
        self.use_psa = use_psa
        if self.use_psa:
            self.attn = SP_SA(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                              proj_drop=drop)
        else:
            self.attn = Attention(
                dim, num_heads=num_heads, head_dim=head_dim, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=self.mlp_hidden_dim, act_layer=act_layer, drop=drop,
                       group=group)
        self.pos_block = PosCNN(dim, dim, 1)

    def forward(self, x, padding_mask=None):
        B, N, C = x.shape
        x = x + self.drop_path(self.attn(self.norm1(x), padding_mask)) / self.skip_lam
        x = x + self.drop_path(self.mlp(self.norm2(x))) / self.skip_lam
        x = self.pos_block(x, int(N ** 0.5), int(N ** 0.5))
        return x

    def flops(self, s):
        heads = self.attn.num_heads
        h = self.dim
        i = self.mlp_hidden_dim
        mha_block_flops = dict(
            kqv=3 * h * h,
            attention_scores=h * s,
            attn_softmax=SOFTMAX_FLOPS * s * heads,
            attention_dropout=DROPOUT_FLOPS * s * heads,
            attention_scale=s * heads,
            attention_weighted_avg_values=h * s,
            attn_output=h * h,
            attn_output_bias=h,
            attn_output_dropout=DROPOUT_FLOPS * h,
            attn_output_residual=h,
            attn_output_layer_norm=LAYER_NORM_FLOPS * h, )

        rpe_flops = dict(
            attention_scores=h * s,
            # attn_softmax=SOFTMAX_FLOPS * s * heads,
            # attention_dropout=DROPOUT_FLOPS * s * heads,
            # attention_scale=s * heads,
            # attention_weighted_avg_values=h * s,
            attn_output=h * h,
            # attn_output_bias=h,
            # attn_output_dropout=DROPOUT_FLOPS * h,
            # attn_output_residual=h,
            # attn_output_layer_norm=LAYER_NORM_FLOPS * h,
            )

        spsa_block_flops = dict(
            w1=2 * h * (4 * s - 4 * s ** 0.5) * 1 / s,
            w2=h * (4 * s - 4 * s ** 0.5) * 1 / s,
            act=h * (4 * s - 4 * s ** 0.5) * 1 / s,
            b1=h * (4 * s - 4 * s ** 0.5) * 1 / s,
            b2=heads * (4 * s - 4 * s ** 0.5) * 1 / s,
            multi=s * heads,
            kqv=3 * h * h,
            attention_scores=h * s,
            attn_softmax=SOFTMAX_FLOPS * s * heads,
            attention_dropout=DROPOUT_FLOPS * s * heads,
            attention_scale=s * heads,
            attention_weighted_avg_values=h * s,
            attn_output=h * h,
            attn_output_bias=h,
            attn_output_dropout=DROPOUT_FLOPS * h,
            attn_output_residual=h,
            attn_output_layer_norm=LAYER_NORM_FLOPS * h, )

        poscnn_block_flops = dict(
            dconv=h * 9 + h,
            res=h,
        )

        ffn_block_flops = dict(
            intermediate=h * i,
            intermediate_act=ACTIVATION_FLOPS * i,
            intermediate_bias=i,
            output=h * i,
            output_bias=h,
            output_dropout=DROPOUT_FLOPS * h,
            output_residual=h,
            output_layer_norm=LAYER_NORM_FLOPS * h, )


        # if self.use_psa:
        #     return sum(spsa_block_flops.values()) * s + sum(ffn_block_flops.values()) * s + sum(
        #         poscnn_block_flops.values()) * s
        # else:
        #     return sum(mha_block_flops.values()) * s + sum(ffn_block_flops.values()) * s + sum(
        #         poscnn_block_flops.values()) * s

        # for our skeleton transformer
        # return sum(mha_block_flops.values()) * s + sum(ffn_block_flops.values()) * s + 5*h*h*s
        return sum(mha_block_flops.values()) * s + sum(rpe_flops.values())*s + h*h*s + h*h*s + 2*h*h*s

class MHABlock(nn.Module):
    """
    Multihead Attention block with residual branch
    """

    def __init__(self, dim, num_heads, head_dim=None, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, group=1, skip_lam=1.):
        super().__init__()
        self.dim = dim
        self.norm1 = norm_layer(dim)
        self.skip_lam = skip_lam
        self.attn = Attention(
            dim, num_heads=num_heads, head_dim=head_dim, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
            proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, padding_mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x * self.skip_lam), padding_mask)) / self.skip_lam
        return x

    def flops(self, s):
        heads = self.attn.num_heads
        h = self.dim
        block_flops = dict(
            kqv=3 * h * h,
            attention_scores=h * s,
            attn_softmax=SOFTMAX_FLOPS * s * heads,
            attention_dropout=DROPOUT_FLOPS * s * heads,
            attention_scale=s * heads,
            attention_weighted_avg_values=h * s,
            attn_output=h * h,
            attn_output_bias=h,
            attn_output_dropout=DROPOUT_FLOPS * h,
            attn_output_residual=h,
            attn_output_layer_norm=LAYER_NORM_FLOPS * h, )

        return sum(block_flops.values()) * s


class FFNBlock(nn.Module):
    """
    Feed forward network with residual branch
    """

    def __init__(self, dim, num_heads, head_dim=None, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, group=1, skip_lam=1.):
        super().__init__()
        self.skip_lam = skip_lam
        self.dim = dim
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=self.mlp_hidden_dim, act_layer=act_layer, drop=drop,
                       group=group)

    def forward(self, x):
        x = x + self.drop_path(self.mlp(self.norm2(x * self.skip_lam))) / self.skip_lam
        return x

    def flops(self, s):
        heads = self.attn.num_heads
        h = self.dim
        i = self.mlp_hidden_dim
        block_flops = dict(
            intermediate=h * i,
            intermediate_act=ACTIVATION_FLOPS * i,
            intermediate_bias=i,
            output=h * i,
            output_bias=h,
            output_dropout=DROPOUT_FLOPS * h,
            output_residual=h,
            output_layer_norm=LAYER_NORM_FLOPS * h, )

        return sum(block_flops.values()) * s


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    """

    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Conv2d(feature_dim, embed_dim, kernel_size=1)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = self.proj(x)
        return x


class PatchEmbedNaive(nn.Module):
    """
    Image to Patch Embedding
    from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        return x

    def flops(self):
        img_size = self.img_size[0]
        block_flops = dict(
            proj=img_size * img_size * 3 * self.embed_dim,
        )
        return sum(block_flops.values())


class PatchEmbed4_2(nn.Module):
    """
    Image to Patch Embedding with 4 layer convolution
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        new_patch_size = to_2tuple(patch_size // 2)

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.embed_dim = embed_dim

        self.conv1 = nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 112x112
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)  # 112x112
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)

        self.proj = nn.Conv2d(64, embed_dim, kernel_size=new_patch_size, stride=new_patch_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.proj(x)  # [B, C, W, H]

        return x

    def flops(self):
        img_size = self.img_size[0]
        block_flops = dict(
            conv1=img_size / 2 * img_size / 2 * 3 * 64 * 7 * 7,
            conv2=img_size / 2 * img_size / 2 * 64 * 64 * 3 * 3,
            conv3=img_size / 2 * img_size / 2 * 64 * 64 * 3 * 3,
            proj=img_size / 2 * img_size / 2 * 64 * self.embed_dim,
        )
        return sum(block_flops.values())


class PatchEmbed4_2_128(nn.Module):
    """ 
    Image to Patch Embedding with 4 layer convolution and 128 filters
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        new_patch_size = to_2tuple(patch_size // 2)

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.embed_dim = embed_dim

        self.conv1 = nn.Conv2d(in_chans, 128, kernel_size=7, stride=2, padding=3, bias=False)  # 112x112
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)  # 112x112
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)

        self.proj = nn.Conv2d(128, embed_dim, kernel_size=new_patch_size, stride=new_patch_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.proj(x)  # [B, C, W, H]

        return x

    def flops(self):
        img_size = self.img_size[0]
        block_flops = dict(
            conv1=img_size / 2 * img_size / 2 * 3 * 128 * 7 * 7,
            conv2=img_size / 2 * img_size / 2 * 128 * 128 * 3 * 3,
            conv3=img_size / 2 * img_size / 2 * 128 * 128 * 3 * 3,
            proj=img_size / 2 * img_size / 2 * 128 * self.embed_dim,
        )
        return sum(block_flops.values())