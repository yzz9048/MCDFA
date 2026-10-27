import math
import torch
import numpy as np
import transformers
import collections
import typing
from datetime import datetime
import torch.nn.functional as F

from torch import nn
from einops import rearrange, reduce, repeat
from Models.interpretable_diffusion.model_utils import LearnablePositionalEncoding, Conv_MLP, Linear,\
                                                       AdaLayerNorm, Transpose, GELU2, series_decomp
from Models.interpretable_diffusion.adaptation import Adaptation
from Models.interpretable_diffusion import down_up
from Models.interpretable_diffusion.Granger import CausalCrossAttention
from Models.interpretable_diffusion.history_drift import HistoryDriftAdapter
def clip(x, max_norm=1):
        x_norm = torch.norm(x, dim=-1, keepdim=True)
        scale = torch.clip(max_norm / x_norm, max=1)
        return x * scale, x_norm

class TrendBlock(nn.Module):
    """
    Model trend of time series using the polynomial regressor.
    """
    def __init__(self, in_dim, out_dim, in_feat, out_feat, act):
        super(TrendBlock, self).__init__()
        trend_poly = 3
        self.trend = nn.Sequential(
            nn.Conv1d(in_channels=in_dim, out_channels=trend_poly, kernel_size=3, padding=1),
            act,
            Transpose(shape=(1, 2)),
            nn.Conv1d(in_feat, out_feat, 3, stride=1, padding=1)
        )

        lin_space = torch.arange(1, out_dim + 1, 1) / (out_dim + 1)
        self.poly_space = torch.stack([lin_space ** float(p + 1) for p in range(trend_poly)], dim=0)

    def forward(self, input,shape):
        b, c, h = input.shape
        x = self.trend(input).transpose(1, 2)
        trend_vals = torch.matmul(x.transpose(1, 2), self.poly_space.to(x.device))
        trend_vals = trend_vals.transpose(1, 2)
        return trend_vals
    

class MovingBlock(nn.Module):
    """
    Model trend of time series using the moving average.
    """
    def __init__(self, out_dim):
        super(MovingBlock, self).__init__()
        size = max(min(int(out_dim / 4), 24), 4)
        self.decomp = series_decomp(size)

    def forward(self, input):
        b, c, h = input.shape
        x, trend_vals = self.decomp(input)
        return x, trend_vals


class FourierLayer(nn.Module):
    """
    Model seasonality of time series using the inverse DFT.
    """
    def __init__(self, d_model, low_freq=1, factor=1):
        super().__init__()
        self.d_model = d_model
        self.factor = factor
        self.low_freq = low_freq
        
    def forward(self, x, shape):
        """x: (b, t, d), shape: 目标时间长度（如 10）"""
        b, t_input, d = x.shape  # [128, 30, 64]
        t_target = shape  # 10

        # 计算原始频谱（基于输入长度 t_input=30）
        x_freq = torch.fft.rfft(x, dim=1)  # [128, 16, 64] (因为 30//2 +1 =16)

        # 选择低频部分（跳过最低频率 self.low_freq=1）
        if t_input % 2 == 0:
            x_freq = x_freq[:, self.low_freq:-1]  # [128, 14, 64] (16-2=14)
            f = torch.fft.rfftfreq(t_input)[self.low_freq:-1]  # [14]
        else:
            x_freq = x_freq[:, self.low_freq:]  # [128, 15, 64] (16-1=15)
            f = torch.fft.rfftfreq(t_input)[self.low_freq:]  # [15]

        # 选取 TopK 频率（假设 self.topk_freq 返回 [128, k, 64] 和索引）
        x_freq, index_tuple = self.topk_freq(x_freq)  # x_freq.shape = [128, k, 64]

        # 调整频率 f 的维度以匹配 x_freq
        f = f.to(x_freq.device)  # [k] 或 [14]/[15] -> 需按 index_tuple 筛选
        f = repeat(f, 'f -> b f d', b=x_freq.size(0), d=x_freq.size(2))  # [128, k, 64]
        f = f[index_tuple]  # 按 topk 索引筛选
        f = rearrange(f, 'b f d -> b f () d')  # [128, k, 1, 64]

        # 外推至 t_target=10
        return self.extrapolate(x_freq, f, t_target)

    def extrapolate(self, x_freq, f, t):
        x_freq = torch.cat([x_freq, x_freq.conj()], dim=1)
        f = torch.cat([f, -f], dim=1)
        t = rearrange(torch.arange(t, dtype=torch.float),
                      't -> () () t ()').to(x_freq.device)

        amp = rearrange(x_freq.abs(), 'b f d -> b f () d')
        phase = rearrange(x_freq.angle(), 'b f d -> b f () d')
        x_time = amp * torch.cos(2 * math.pi * f * t + phase)
        return reduce(x_time, 'b f t d -> b t d', 'sum')

    def topk_freq(self, x_freq):
        length = x_freq.shape[1]
        top_k = int(self.factor * math.log(length)) #对序列数量取ln
        values, indices = torch.topk(x_freq.abs(), top_k, dim=1, largest=True, sorted=True)
        mesh_a, mesh_b = torch.meshgrid(torch.arange(x_freq.size(0)), torch.arange(x_freq.size(2)), indexing='ij')
        index_tuple = (mesh_a.unsqueeze(1), indices, mesh_b.unsqueeze(1))
        x_freq = x_freq[index_tuple]
        return x_freq, index_tuple
    

class SeasonBlock(nn.Module):
    """
    Model seasonality of time series using the Fourier series.
    """
    def __init__(self, in_dim, out_dim, factor=1):
        super(SeasonBlock, self).__init__()
        season_poly = factor * min(32, int(out_dim // 2))
        self.season = nn.Conv1d(in_channels=in_dim, out_channels=season_poly, kernel_size=1, padding=0)
        fourier_space = torch.arange(0, out_dim, 1) / out_dim
        p1, p2 = (season_poly // 2, season_poly // 2) if season_poly % 2 == 0 \
            else (season_poly // 2, season_poly // 2 + 1)
        s1 = torch.stack([torch.cos(2 * np.pi * p * fourier_space) for p in range(1, p1 + 1)], dim=0)
        s2 = torch.stack([torch.sin(2 * np.pi * p * fourier_space) for p in range(1, p2 + 1)], dim=0)
        self.poly_space = torch.cat([s1, s2])

    def forward(self, input):
        b, c, h = input.shape
        x = self.season(input)
        season_vals = torch.matmul(x.transpose(1, 2), self.poly_space.to(x.device))
        season_vals = season_vals.transpose(1, 2)
        return season_vals


class FullAttention(nn.Module):
    def __init__(self,
                 n_embd, # the embed dim
                 n_head, # the number of heads
                 attn_pdrop=0.1, # attention dropout prob
                 resid_pdrop=0.1, # residual attention dropout prob
    ):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x, mask=None):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, T)

        att = F.softmax(att, dim=-1) # (B, nh, T, T)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side, (B, T, C)
        att = att.mean(dim=1, keepdim=False) # (B, T, T)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, att


class CrossAttention(nn.Module):
    def __init__(self,
                 n_embd, # the embed dim
                 condition_embd, # condition dim
                 n_head, # the number of heads
                 attn_pdrop=0.1, # attention dropout prob
                 resid_pdrop=0.1, # residual attention dropout prob
    ):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(condition_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(condition_embd, n_embd)
        
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x, encoder_output, mask=None):
        B, T, C = x.size()
        B, T_E, _ = encoder_output.size()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(encoder_output).view(B, T_E, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(encoder_output).view(B, T_E, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, T)

        att = F.softmax(att, dim=-1) # (B, nh, T, T)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side, (B, T, C)
        att = att.mean(dim=1, keepdim=False) # (B, T, T)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, att
    

class EncoderBlock(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self,
                 n_embd=1024,
                 n_head=16,
                 attn_pdrop=0.1,
                 resid_pdrop=0.1,
                 mlp_hidden_times=4,
                 activate='GELU'
                 ):
        super().__init__()

        self.ln1 = AdaLayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = FullAttention(
                n_embd=n_embd,
                n_head=n_head,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
            )
        
        assert activate in ['GELU', 'GELU2']
        act = nn.GELU() if activate == 'GELU' else GELU2()

        self.mlp = nn.Sequential(
                nn.Linear(n_embd, mlp_hidden_times * n_embd),
                act,
                nn.Linear(mlp_hidden_times * n_embd, n_embd),
                nn.Dropout(resid_pdrop),
            )
        
    def forward(self, x, timestep, mask=None, label_emb=None):
        a, att = self.attn(self.ln1(x, timestep, label_emb), mask=mask)
        x = x + a
        x = x + self.mlp(self.ln2(x))   # only one really use encoder_output
        return x, att


class Encoder(nn.Module):
    def __init__(
        self,
        n_layer=14,
        n_embd=1024,
        n_head=16,
        attn_pdrop=0.1, # 由0变0.1
        resid_pdrop=0.,
        mlp_hidden_times=4,
        block_activate='GELU',
    ):
        super().__init__()
        
        self.n_embd = n_embd
        self.blocks = nn.Sequential(*[EncoderBlock(
                n_embd=n_embd,
                n_head=n_head,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
                mlp_hidden_times=mlp_hidden_times,
                activate=block_activate,
        ) for _ in range(n_layer)])

    def forward(self, input, t, padding_masks=None, label_emb=None):
        x = input
        for block_idx in range(len(self.blocks)):
            x, _ = self.blocks[block_idx](x, t, mask=padding_masks, label_emb=label_emb)
        return x

class FullyConnected(nn.Module):
    def __init__(self, in_feat, out_dim, linear_sizes):
        super(FullyConnected, self).__init__()
        # 处理每个 [features] 的子网络
        layers = []
        for i, hidden in enumerate(linear_sizes):
            prev = in_feat if i == 0 else linear_sizes[i - 1]
            layers += [nn.Linear(prev, hidden), nn.ReLU()]
        self.feature_net = nn.Sequential(*layers)
        
        # 输出层：基于 window 的聚合特征（例如平均池化）输出二分类
        self.classifier = nn.Linear(linear_sizes[-1], out_dim)

    def forward(self, x: torch.Tensor):  # x: [B, W, F]
        B, W, F = x.shape
        # reshape to [B*W, F] to apply feature_net per window
        x = x.view(B * W, F)
        x = self.feature_net(x)  # [B*W, hidden]
        x = x.view(B, W, -1)     # reshape back to [B, W, hidden]

        # 聚合窗口（例如平均池化）
        x = x.mean(dim=1)        # [B, hidden]

        return self.classifier(x)  # [B, 2]

class AttentionFC(nn.Module):
    def __init__(self, in_feat, out_dim, linear_sizes):
        super(AttentionFC, self).__init__()

        # 局部特征处理：对每个 [features] 输入进行处理
        layers = []
        for i, hidden in enumerate(linear_sizes):
            prev = in_feat if i == 0 else linear_sizes[i - 1]
            layers += [nn.Linear(prev, hidden), nn.ReLU()]
        self.feature_net = nn.Sequential(*layers)

        self.hidden_dim = linear_sizes[-1]

        # 注意力权重层：投影成 [B, W, 1]，对 window 加权
        self.attn_fc = nn.Linear(self.hidden_dim, 1)

        # 最终分类器
        self.classifier = nn.Linear(self.hidden_dim, out_dim)

    def forward(self, x: torch.Tensor):  # x: [B, W, F]
        B, W, F = x.shape

        # 展平窗口以便 feature_net 批处理
        x = x.view(B * W, F)
        x = self.feature_net(x)  # [B*W, hidden]
        x = x.view(B, W, -1)     # [B, W, hidden]

        # 注意力权重计算
        attn_weights = self.attn_fc(x)  # [B, W, 1]
        attn_weights = F.softmax(attn_weights, dim=1)  # [B, W, 1]

        # 加权求和
        x = (x * attn_weights).sum(dim=1)  # [B, hidden]

        # 分类输出
        return self.classifier(x)  # [B, 2]

class DecoderBlock(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self,
                 n_channel,
                 n_shape,
                 n_feat,
                 n_embd=1024,
                 n_head=16,
                 attn_pdrop=0.1,
                 resid_pdrop=0.1,
                 mlp_hidden_times=4,
                 activate='GELU',
                 condition_dim=1024,
                 ):
        super().__init__()
        
        self.ln1 = AdaLayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        self.attn1 = FullAttention(
                n_embd=n_embd,
                n_head=n_head,
                attn_pdrop=attn_pdrop, 
                resid_pdrop=resid_pdrop,
                )
        self.attn2 = CrossAttention(
                n_embd=n_embd,
                condition_embd=condition_dim,
                n_head=n_head,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
                )
        
        self.ln1_1 = AdaLayerNorm(n_embd)

        assert activate in ['GELU', 'GELU2']
        act = nn.GELU() if activate == 'GELU' else GELU2()
        self.n_shape = n_shape
        self.trend = TrendBlock(n_channel, n_shape, n_embd, n_embd, act=act)
        # self.decomp = MovingBlock(n_channel)
        self.seasonal = FourierLayer(d_model=n_embd)
        # self.seasonal = SeasonBlock(n_channel, n_channel)

        self.mlp = nn.Sequential(
            nn.Linear(n_embd, mlp_hidden_times * n_embd),
            act,
            nn.Linear(mlp_hidden_times * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

        self.proj = nn.Conv1d(n_shape, n_channel * 2, 1)
        self.linear = nn.Linear(n_embd, n_feat-2)

    def forward(self, x, encoder_output, timestep, mask=None, label_emb=None):
        a, att = self.attn1(self.ln1(x, timestep, label_emb), mask=mask)
        x = x + a
        a, att = self.attn2(self.ln1_1(x, timestep), encoder_output, mask=mask)
        x = x + a
        x1, x2 = self.proj(x).chunk(2, dim=1)
        trend, season = self.trend(x1,self.n_shape), self.seasonal(x2,self.n_shape)
        x = self.mlp(self.ln2(x))
        return x, trend, season
    

class Decoder(nn.Module):
    def __init__(
        self,
        n_channel,
        n_shape,
        n_feat,
        n_embd=1024,
        n_head=16,
        n_layer=10,
        attn_pdrop=0.1,
        resid_pdrop=0.1,
        mlp_hidden_times=4,
        block_activate='GELU',
        condition_dim=512    
    ):
        super().__init__()
        self.d_model = n_embd
        self.n_feat = n_feat
        self.n_shape = n_shape
        
        self.blocks = nn.ModuleList([
            DecoderBlock(
                n_feat=n_feat,
                n_channel=n_channel,
                n_shape=n_shape,
                n_embd=n_embd,
                n_head=n_head,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
                mlp_hidden_times=mlp_hidden_times,
                activate=block_activate,
                condition_dim=condition_dim,
            ) for _ in range(n_layer)
        ])

    def forward(self, x, t, enc, padding_masks=None, label_emb=None):
        b, _, _ = x.shape
        c = self.n_shape

        # 存储所有层的趋势和季节
        total_trend = torch.zeros((b, c, self.d_model), device=x.device)
        total_season = torch.zeros((b, c, self.d_model), device=x.device)

        residual = x  # 初始残差为输入

        for block in self.blocks:
            # 每个 block 对当前残差拟合
            residual_out, residual_trend, residual_season = \
                block(residual, enc, t, mask=padding_masks, label_emb=label_emb)
            
            # 累积趋势与季节
            total_trend += residual_trend
            total_season += residual_season
            
            # 更新残差 (严格 N-Beats 递推)
            residual = residual_out - (residual_trend + residual_season)

        return residual, total_trend, total_season

class Bottleneck(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, n_layers: int,
                 bottleneck_dim: typing.Union[typing.List[int], int], activation=nn.LeakyReLU, need_bias: bool = False,
                 shared=False, rand_init: dict = None):
        super().__init__()
        self.out_dim = out_dim
        self.activation = activation()
        self.shared = shared
        self.need_bias = need_bias
        self.biases = nn.ParameterList([nn.Parameter(torch.empty(n_layers, 1, bottleneck_dim))])
        self.weights = nn.ParameterList([nn.Linear(in_dim, bottleneck_dim, bias=False) if self.shared else
                                         nn.Parameter(torch.empty(n_layers, 1, bottleneck_dim, in_dim))])
        if self.need_bias:
            self.biases.append(nn.Parameter(torch.zeros(n_layers, 1, out_dim)))
        self.weights.append(nn.Parameter(torch.zeros(1 if self.shared else n_layers, 1, out_dim, bottleneck_dim)))
        for i in range(0, len(self.weights) - 1):
            for j in range(n_layers):
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
                    self.weights[i].weight if shared else self.weights[i][0, 0])
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.biases[i][j], -bound, bound)
                if not self.shared:
                    nn.init.kaiming_uniform_(self.weights[i][j, 0], a=math.sqrt(5))
        if self.need_bias:
            nn.init.zeros_(self.biases[-1])
            if rand_init is not None:
                for pos, (r, in_fea) in rand_init.items():
                    nn.init.kaiming_uniform_(self.biases[-1][pos, :, :r * in_fea], a=math.sqrt(5))

        self.memories = None
        self.last_adaptation = None
    
    def forward(self, x, mask=None, training=False):
        if self.shared:
            x = self.activation(self.weights[0](x) + self.biases[0]).unsqueeze(-1)
        else:
            x = self.activation(self.weights[0] @ x.unsqueeze(-1) + self.biases[0].unsqueeze(-1))
        x = self.weights[-1] @ x
        x = x.squeeze(-1)
        if self.need_bias:
            x = x + self.biases[-1]
        return x

class AdaptGenerator(nn.Module):
    def __init__(self, backbone: nn.Module, concept_features: int, n_shape: int, n_dim: int,activation,
                 shared: bool = True, need_bias: bool = True, adaptive_dim: bool = False, mid_dim: int = None):
        super().__init__()
        self.dim_name_dict = collections.defaultdict(list)
        self.bottlenecks = nn.ModuleDict()
        self.loras = nn.ModuleDict()
        for name, module in backbone.named_modules():
            if name == "emb_judge":
                self.dim_name_dict[name + '_' + str(n_shape*2+n_dim)].append(name)
                break
        for key, names in self.dim_name_dict.items():
            out_dim = int(key.split('_')[-1])
            self.bottlenecks[key] = Bottleneck(concept_features, out_dim, len(names), mid_dim,
                                               activation, shared=shared, need_bias=need_bias)

    def forward(self, x, need_clip=False):
        if need_clip:
            x, x_norm = clip(x)
        coefs = {k: bottleneck(x) for k, bottleneck in self.bottlenecks.items()}
        return coefs    

class Transformer(nn.Module):
    def __init__(
        self,
        n_feat,
        n_channel,
        n_shape,
        n_layer_enc=5,
        n_layer_dec=14,
        n_embd=1024,
        n_heads=16,
        attn_pdrop=0.1,
        resid_pdrop=0.1,
        mlp_hidden_times=4,
        block_activate='GELU',
        max_len=2048,
        conv_params=None,
        drift_adaptation=None,
        causal_mask_threshold=None,
        **kwargs
    ):
        super().__init__()
        drift_config = dict(drift_adaptation or {})
        self.history_adaptation_enabled = bool(drift_config.pop('enabled', False))
        # Keep historical defaults; an explicit threshold also enables the mask
        # in the history drift path used by the threshold sweep.
        self.mask_history_scores = causal_mask_threshold is not None
        self.causal_mask_threshold = 0.1 if causal_mask_threshold is None else float(causal_mask_threshold)
        if not math.isfinite(self.causal_mask_threshold) or not 0 <= self.causal_mask_threshold <= 1:
            raise ValueError('causal_mask_threshold must be finite and in [0, 1].')
        self.emb = Conv_MLP(n_feat, n_embd, resid_pdrop=resid_pdrop)
        self.emb2 = Conv_MLP(n_feat-2, n_embd, resid_pdrop=resid_pdrop)
        self.emb_cond = Conv_MLP(n_feat-2, n_embd, resid_pdrop=resid_pdrop)
        self.emb_cond2 = Conv_MLP(2, n_embd, resid_pdrop=resid_pdrop)
        self.emb_judge = Linear(n_feat-2, n_embd, n_shape,
                                consistent_inference=self.history_adaptation_enabled)
        self.inverse = Conv_MLP(n_embd, n_feat-2, resid_pdrop=resid_pdrop)
        self.grangerLag = n_channel - n_shape if n_shape<n_channel else n_channel

        self.li = nn.Linear(n_channel - n_shape, n_embd) if self.history_adaptation_enabled else nn.Linear(20, 64)

        if conv_params is None or conv_params[0] is None:
            if n_feat < 32 and n_channel < 64:
                kernel_size, padding = 1, 0
            else:
                kernel_size, padding = 5, 2
        else:
            kernel_size, padding = conv_params

        self.combine_s = nn.Conv1d(n_embd, n_feat-2, kernel_size=kernel_size, stride=1, padding=padding,
                                   padding_mode='circular', bias=False)
        self.combine_t = nn.Conv1d(n_embd, n_feat-2, kernel_size=kernel_size, stride=1, padding=padding,
                                   padding_mode='circular', bias=False)
        self.dropout_s = nn.Dropout(p=resid_pdrop) # 新增
        self.combine_m = nn.Conv1d(n_layer_dec, 1, kernel_size=1, stride=1, padding=0,
                                   padding_mode='circular', bias=False)
        self.dropout_m = nn.Dropout(p=resid_pdrop) # 新增

        self.encoder = Encoder(n_layer_enc, n_embd, n_heads, attn_pdrop, resid_pdrop, mlp_hidden_times, block_activate)
        self.encoder2 = Encoder(n_layer_enc, n_embd, n_heads, attn_pdrop, resid_pdrop, mlp_hidden_times, block_activate)

        self.pos_enc = LearnablePositionalEncoding(n_embd, dropout=resid_pdrop, max_len=max_len-n_shape if n_shape<max_len else max_len)
        self.pos_enc2 = LearnablePositionalEncoding(n_embd, dropout=resid_pdrop, max_len=max_len-n_shape if n_shape<max_len else max_len)

        self.detecter = FullyConnected(n_embd if self.history_adaptation_enabled else 64, 2, [64,32])
        self.decoder = Decoder(n_channel,n_shape,n_feat, n_embd, n_heads, n_layer_dec, attn_pdrop, resid_pdrop, mlp_hidden_times,
                               block_activate, condition_dim=n_embd)
        self.pos_dec = LearnablePositionalEncoding(n_embd, dropout=resid_pdrop, max_len=n_shape)
        self.pos_dec2 = LearnablePositionalEncoding(n_embd, dropout=resid_pdrop, max_len=n_channel-n_shape)
        self.pos_dec3 = LearnablePositionalEncoding(n_embd, dropout=resid_pdrop, max_len=n_shape)

        if self.history_adaptation_enabled:
            self.drift_adapter = HistoryDriftAdapter(n_feat - 1, n_shape, n_embd, **drift_config)
        else:
            self.mlp1 = nn.Sequential(Transpose(shape=(1, 2)), nn.Linear(n_shape, 200), nn.GELU(),
                                      nn.Linear(200, 200))
            self.mlp2 = nn.Sequential(Transpose(shape=(1, 2)), nn.Linear(n_channel - n_shape, 200), nn.GELU(),
                                      nn.Linear(200, 200))
            self.generator = AdaptGenerator(self, 200, n_shape, n_embd, nn.Sigmoid, True, True, False, 32)

        self.atte1 = CausalCrossAttention(num_features=n_feat-2, b_dim=1)
        self.atte2 = CausalCrossAttention(num_features=n_feat-2, b_dim=1)
    def reset_drift_state(self):
        if self.history_adaptation_enabled:
            self.drift_adapter.reset_state()

    def prepare_history(self, history, granger, update_memory=True):
        """Compute the observed condition and adaptation once, before diffusion."""
        if not self.history_adaptation_enabled:
            raise RuntimeError('History caching requires drift_adaptation.enabled.')
        rps = history[:, :, :1]
        gradient = torch.cat([torch.zeros_like(rps[:, :1]), torch.diff(rps, dim=1)], dim=1)
        causal_loss, causal_scores = granger(history[:, :, 1:], rps)
        scores = causal_scores.detach()
        if self.mask_history_scores:
            mask = scores > self.causal_mask_threshold
            scores = scores * mask
        cross_raw, raw_gate = self.atte1(history[:, :, 1:], rps)
        cross_grad, grad_gate = self.atte2(history[:, :, 1:], gradient)
        cross_raw = history[:, :, 1:] + raw_gate * (cross_raw - history[:, :, 1:])
        cross_grad = history[:, :, 1:] + grad_gate * (cross_grad - history[:, :, 1:])
        condition = (1 - scores.unsqueeze(1)) * history[:, :, 1:] + scores.unsqueeze(1) * (cross_raw + cross_grad) * 0.5
        condition_embedding = self.pos_enc(self.emb_cond(condition))
        # The historical judge representation has no diffusion timestep/noise dependency.
        zero_step = torch.zeros(len(history), dtype=torch.long, device=history.device)
        judge_history = self.encoder(condition_embedding, zero_step)
        adaptation = self.drift_adapter(history, granger.last_full_error, scores, update_memory=update_memory)
        return {'condition_embedding': condition_embedding, 'judge_history': judge_history,
                'adaptation': adaptation, 'causal_loss': causal_loss,
                'observed_metric_scale': history[:, :, 1:].detach().abs().amax(dim=1, keepdim=True).clamp_min(1)}

    def _forward_history(self, input, history, t, granger, padding_masks, context):
        context = self.prepare_history(history, granger) if context is None else context
        padded = torch.cat([input.new_zeros(input.shape[0], input.shape[1], 2), input], dim=-1)
        inp_dec = self.pos_dec(self.emb(padded))
        enc_cond = self.encoder(context['condition_embedding'], t, padding_masks=padding_masks)
        output, trend, season = self.decoder(inp_dec, t, enc_cond, padding_masks=padding_masks)
        self.emb_judge.assign_adaptation(context['adaptation'])
        forecast_judge = self.emb_judge(trend + season + output)
        history_judge = self.li(context['judge_history'].transpose(1, 2)).transpose(1, 2)
        judge = self.detecter(torch.cat([history_judge, forecast_judge], dim=1))
        return self.dropout_m(self.combine_t((trend + output).transpose(1, 2)).transpose(1, 2)), \
            self.dropout_s(self.combine_s(season.transpose(1, 2)).transpose(1, 2)), judge, context['causal_loss']

    def forward(self, input, history, t, clip, granger, padding_masks=None, return_res=False,
                history_context=None):
        if self.history_adaptation_enabled:
            return self._forward_history(input, history, t, granger, padding_masks, history_context)
        # start2 = datetime.now()
        # print("causal start at :",start2)

        rps_raw_data = history[:,:,0]
        rps_raw_grad = torch.diff(rps_raw_data, dim=0) # 由于rps采样间隔是 1s ，所以一阶梯度即一阶差分
        rps_raw_grad = torch.cat([torch.zeros(1, rps_raw_data.shape[1]).cuda(), rps_raw_grad], dim=0)
        rps_data = torch.cat([rps_raw_data.unsqueeze(-1),rps_raw_grad.unsqueeze(-1)],dim=-1) # 【batch_size, window-DeltaT,2】

        cross_data1, _ = self.atte1(history[:,:,1:], rps_raw_data.unsqueeze(-1))
        cross_data2, _ = self.atte2(history[:,:,1:], rps_raw_grad.unsqueeze(-1))  
        cross_data = 1/2*cross_data1 + 1/2*cross_data2  
        loss_causal, causal_scores = granger(history[:,:,1:], rps_data[:,:,0].unsqueeze(-1))

        mask = causal_scores > self.causal_mask_threshold
        causal_scores = causal_scores * mask
        
        cond_data = (1 - causal_scores.detach().unsqueeze(1)) * history[:,:,1:] + causal_scores.detach().unsqueeze(1) * cross_data
        
        # end2 = datetime.now()
        # print("causal end at :",end2)

        # 融合噪声和条件数据作为输入
        input = torch.cat([torch.zeros(input.shape[0], input.shape[1], 2).cuda(),input], dim=-1)
        emb = self.emb(input) # 输入是对target的加噪数据 
        inp_dec = self.pos_dec(emb)
        cond_feature = self.emb_cond(cond_data)
        inp_enc = self.pos_enc(cond_feature)
        enc_cond = self.encoder(inp_enc, t, padding_masks=padding_masks)
        
        output, trend, season = self.decoder(inp_dec, t, enc_cond, padding_masks=padding_masks)

        # end3 = datetime.now()
        # print("condition end at :",end3)
        
        # 仅需对预测的输出（检测的输入）做漂移适应
        emb2 = self.emb2(cond_data) 
        inp_dec2 = self.pos_dec2(emb2)
        concept = self.mlp1(self.pos_dec3(trend+season+output)).mean(-2)
        recent_concept = self.mlp2(inp_dec2).mean(-2).mean(list(range(0, output.dim() - 2)))
        drift = concept - recent_concept
        adaptations = self.generator(drift, need_clip=clip)
        for out_dim, adaptation in adaptations.items():
            for i in range(len(adaptation)):
                name = self.generator.dim_name_dict[out_dim][i]
                self.get_submodule(name).assign_adaptation(adaptation[i])

        outForJudge = self.emb_judge(trend+season+output)
        # self.li换成 卷积如何
        judge = self.detecter(torch.cat([self.li(enc_cond.transpose(1, 2)).transpose(1, 2),outForJudge],dim=1))

        # end4 = datetime.now()
        # print("judge end at :",end4)

        return self.dropout_m(self.combine_t((trend+output).transpose(1, 2)).transpose(1, 2)), \
            self.dropout_s(self.combine_s(season.transpose(1, 2)).transpose(1, 2)), judge, loss_causal

    def _ssf_input(self, x: torch.Tensor):
        batch_size = x.size()[:-1]
        x = x.reshape(self.scale2.shape[0], -1, x.shape[-1])
        return (x * self.scale2).view(*batch_size, x.shape[-1])
    
