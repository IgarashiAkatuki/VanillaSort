import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 50000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # [1, L, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class ConvFrontend(nn.Module):
    """
    轻量时间前端：
    - 先在时间轴上做局部卷积，提取 spike 的瞬时波形模式
    - 再映射到 Transformer 的 d_model
    不使用 channel positional encoding。
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        d_model: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2, groups=1, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=4, dilation=2, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
            nn.Conv1d(hidden_channels, d_model, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C]
        x = x.transpose(1, 2)          # [B, C, L]
        x = self.net(x)                # [B, D, L]
        x = x.transpose(1, 2)          # [B, L, D]
        x = self.out_norm(x)
        return x


class GroupNormConvFrontend(nn.Module):
    """Baseline receptive field with batch-independent normalization.

    This opt-in frontend is a BatchNorm diagnostic/fix candidate.  The
    historical ``standard`` frontend remains unchanged for checkpoint and API
    compatibility.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        d_model: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        groups = min(8, int(hidden_channels))
        while hidden_channels % groups != 0:
            groups -= 1
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=7, padding=3, bias=False),
            nn.GroupNorm(groups, hidden_channels),
            nn.GELU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2, bias=False),
            nn.GroupNorm(groups, hidden_channels),
            nn.GELU(),
            nn.Conv1d(
                hidden_channels,
                hidden_channels,
                kernel_size=5,
                padding=4,
                dilation=2,
                bias=False,
            ),
            nn.GroupNorm(groups, hidden_channels),
            nn.GELU(),
            nn.Conv1d(hidden_channels, d_model, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x.transpose(1, 2))
        return self.out_norm(x.transpose(1, 2))


class MultiscaleConvFrontend(nn.Module):
    """Parallel short/medium waveform filters followed by temporal fusion."""

    def __init__(self, in_channels: int, hidden_channels: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        widths = [hidden_channels // 3] * 3
        for index in range(hidden_channels - sum(widths)):
            widths[index] += 1
        self.branches = nn.ModuleList(
            nn.Conv1d(in_channels, width, kernel_size=kernel, padding=kernel // 2, bias=False)
            for width, kernel in zip(widths, (3, 7, 15))
        )
        self.fusion = nn.Sequential(
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
            nn.Conv1d(
                hidden_channels, hidden_channels, kernel_size=5,
                padding=4, dilation=2, bias=False,
            ),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
            nn.Conv1d(hidden_channels, d_model, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = torch.cat([branch(x) for branch in self.branches], dim=1)
        x = self.fusion(x).transpose(1, 2)
        return self.out_norm(x)


class _DilatedResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(
                channels, channels, kernel_size=5, padding=2 * dilation,
                dilation=dilation, bias=False,
            ),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class DilatedResidualConvFrontend(nn.Module):
    """Lightweight residual TCN frontend with a 67-sample receptive field."""

    def __init__(self, in_channels: int, hidden_channels: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.input = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[
                _DilatedResidualBlock(hidden_channels, dilation, dropout)
                for dilation in (1, 2, 4, 8)
            ]
        )
        self.projection = nn.Sequential(
            nn.Conv1d(hidden_channels, d_model, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input(x.transpose(1, 2))
        x = self.blocks(x)
        x = self.projection(x).transpose(1, 2)
        return self.out_norm(x)


class MultiscaleDilatedResidualFrontend(nn.Module):
    """Multi-scale waveform filters feeding the same residual dilated context stack."""

    def __init__(self, in_channels: int, hidden_channels: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        widths = [hidden_channels // 3] * 3
        for index in range(hidden_channels - sum(widths)):
            widths[index] += 1
        self.branches = nn.ModuleList(
            nn.Conv1d(in_channels, width, kernel_size=kernel, padding=kernel // 2, bias=False)
            for width, kernel in zip(widths, (3, 7, 15))
        )
        self.input_norm = nn.Sequential(nn.BatchNorm1d(hidden_channels), nn.GELU())
        self.blocks = nn.Sequential(
            *[
                _DilatedResidualBlock(hidden_channels, dilation, dropout)
                for dilation in (1, 2, 4, 8)
            ]
        )
        self.projection = nn.Sequential(
            nn.Conv1d(hidden_channels, d_model, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.input_norm(torch.cat([branch(x) for branch in self.branches], dim=1))
        x = self.blocks(x)
        x = self.projection(x).transpose(1, 2)
        return self.out_norm(x)


class PhysicalChannelPositionEncoder(nn.Module):
    """Encode centered probe-local channel coordinates in physical units.

    The Fourier terms provide an absolute coordinate code *within the local
    patch*.  Probe depth is intentionally not encoded: the prepared IBL API
    exposes patch-centered coordinates, which are transferable to independent
    four-channel recordings and avoid learning an insertion-specific depth
    shortcut.
    """

    def __init__(self, output_dim: int, coordinate_scale_um: float = 20.0):
        super().__init__()
        if coordinate_scale_um <= 0:
            raise ValueError("coordinate_scale_um must be positive")
        self.coordinate_scale_um = float(coordinate_scale_um)
        # x/y plus sin/cos at three spatial frequencies for each axis.
        feature_dim = 2 + 2 * 2 * 3
        self.net = nn.Sequential(
            nn.Linear(feature_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, coordinates_um: torch.Tensor) -> torch.Tensor:
        if coordinates_um.ndim != 3 or coordinates_um.shape[-1] != 2:
            raise ValueError(
                "channel_position must have shape [B, C, 2], got "
                f"{tuple(coordinates_um.shape)}"
            )
        if not torch.isfinite(coordinates_um).all():
            raise ValueError("channel_position contains NaN or Inf")
        normalized = coordinates_um.float() / self.coordinate_scale_um
        features = [normalized]
        for frequency in (1.0, 2.0, 4.0):
            phase = math.pi * normalized / frequency
            features.extend((torch.sin(phase), torch.cos(phase)))
        return self.net(torch.cat(features, dim=-1))


class GeometryAwareConvFrontend(nn.Module):
    """Permutation-invariant four-channel mixer with physical position codes.

    At every time sample each electrode becomes a token containing its voltage
    and physical coordinate. Channel self-attention then models spatial leakage
    patterns before learned-query pooling produces one representation per time
    point. Temporal convolutions retain sample-level output resolution.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        d_model: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        token_dim = max(16, hidden_channels // 2)
        token_dim = int(math.ceil(token_dim / 4) * 4)
        self.in_channels = int(in_channels)
        self.token_dim = token_dim
        self.amplitude_encoder = nn.Sequential(
            nn.Linear(1, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        self.position_encoder = PhysicalChannelPositionEncoder(token_dim)
        self.token_norm = nn.LayerNorm(token_dim)
        self.channel_attention = nn.MultiheadAttention(
            token_dim, num_heads=4, dropout=dropout, batch_first=True
        )
        self.channel_norm = nn.LayerNorm(token_dim)
        self.pool_query = nn.Parameter(torch.zeros(1, 1, token_dim))
        nn.init.normal_(self.pool_query, std=0.02)
        self.channel_pool = nn.MultiheadAttention(
            token_dim, num_heads=4, dropout=dropout, batch_first=True
        )
        self.temporal = nn.Sequential(
            nn.Conv1d(token_dim, hidden_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
            nn.Conv1d(
                hidden_channels,
                hidden_channels,
                kernel_size=5,
                padding=4,
                dilation=2,
                bias=False,
            ),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
            nn.Conv1d(hidden_channels, d_model, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, channel_position: torch.Tensor | None) -> torch.Tensor:
        if channel_position is None:
            raise ValueError("geometry_aware frontend requires channel_position")
        if x.ndim != 3 or x.shape[-1] != self.in_channels:
            raise ValueError(
                f"Expected x [B,L,{self.in_channels}], got {tuple(x.shape)}"
            )
        if channel_position.shape[:2] != (x.shape[0], self.in_channels):
            raise ValueError(
                "channel_position batch/channel dimensions do not match x: "
                f"{tuple(channel_position.shape)} vs {tuple(x.shape)}"
            )
        batch, length, channels = x.shape
        amplitude = self.amplitude_encoder(x.float().unsqueeze(-1))
        position = self.position_encoder(channel_position).to(dtype=amplitude.dtype)
        tokens = self.token_norm(amplitude + position[:, None, :, :])
        tokens = tokens.reshape(batch * length, channels, self.token_dim)
        mixed, _ = self.channel_attention(
            tokens, tokens, tokens, need_weights=False
        )
        tokens = self.channel_norm(tokens + mixed)
        query = self.pool_query.expand(batch * length, -1, -1)
        pooled, _ = self.channel_pool(query, tokens, tokens, need_weights=False)
        pooled = pooled[:, 0].reshape(batch, length, self.token_dim)
        output = self.temporal(pooled.transpose(1, 2)).transpose(1, 2)
        return self.out_norm(output)


class GeometryResidualConvFrontend(nn.Module):
    """Baseline ConvFrontend plus a zero-initialized physical-geometry adapter.

    A shared temporal convolution first extracts a waveform token independently
    on each electrode. Only then are coordinate codes and spatial attention
    applied. The final adapter projection starts at exactly zero, so the model's
    initial function is the proven baseline rather than a destructive channel
    bottleneck.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        d_model: int,
        dropout: float = 0.1,
        use_channel_position: bool = True,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.use_channel_position = bool(use_channel_position)
        token_dim = max(16, hidden_channels // 2)
        token_dim = int(math.ceil(token_dim / 4) * 4)
        self.token_dim = token_dim
        self.base = ConvFrontend(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            d_model=d_model,
            dropout=dropout,
        )
        self.shared_waveform = nn.Sequential(
            nn.Conv1d(1, token_dim, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(token_dim),
            nn.GELU(),
            nn.Conv1d(
                token_dim,
                token_dim,
                kernel_size=5,
                padding=4,
                dilation=2,
                bias=False,
            ),
            nn.BatchNorm1d(token_dim),
            nn.GELU(),
        )
        self.position_encoder = PhysicalChannelPositionEncoder(token_dim)
        self.channel_attention = nn.MultiheadAttention(
            token_dim, num_heads=4, dropout=dropout, batch_first=True
        )
        self.channel_norm = nn.LayerNorm(token_dim)
        self.pool_query = nn.Parameter(torch.zeros(1, 1, token_dim))
        nn.init.normal_(self.pool_query, std=0.02)
        self.channel_pool = nn.MultiheadAttention(
            token_dim, num_heads=4, dropout=dropout, batch_first=True
        )
        self.adapter_projection = nn.Linear(token_dim, d_model)
        nn.init.zeros_(self.adapter_projection.weight)
        nn.init.zeros_(self.adapter_projection.bias)
        self.adapter_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, channel_position: torch.Tensor | None) -> torch.Tensor:
        if channel_position is None:
            raise ValueError("geometry_residual frontend requires channel_position")
        if x.ndim != 3 or x.shape[-1] != self.in_channels:
            raise ValueError(
                f"Expected x [B,L,{self.in_channels}], got {tuple(x.shape)}"
            )
        if channel_position.shape != (x.shape[0], self.in_channels, 2):
            raise ValueError(
                "channel_position must be [B,C,2] and match x, got "
                f"{tuple(channel_position.shape)}"
            )
        base = self.base(x)
        batch, length, channels = x.shape
        waveforms = self.shared_waveform(
            x.transpose(1, 2).reshape(batch * channels, 1, length)
        )
        waveforms = waveforms.reshape(
            batch, channels, self.token_dim, length
        ).permute(0, 3, 1, 2)
        if self.use_channel_position:
            positions = self.position_encoder(channel_position).to(waveforms.dtype)
        else:
            # Keep the otherwise identical spatial adapter as a matched control:
            # only the physical-coordinate evidence is removed.
            positions = torch.zeros(
                batch,
                channels,
                self.token_dim,
                device=waveforms.device,
                dtype=waveforms.dtype,
            )
        tokens = (waveforms + positions[:, None]).reshape(
            batch * length, channels, self.token_dim
        )
        mixed, _ = self.channel_attention(tokens, tokens, tokens, need_weights=False)
        tokens = self.channel_norm(tokens + mixed)
        query = self.pool_query.expand(batch * length, -1, -1)
        pooled, _ = self.channel_pool(query, tokens, tokens, need_weights=False)
        adapter = self.adapter_projection(pooled[:, 0]).reshape(batch, length, -1)
        return base + self.adapter_dropout(adapter)


class GeometryResidualNoPositionConvFrontend(GeometryResidualConvFrontend):
    """Parameter-matched geometry adapter ablation without coordinate evidence."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        d_model: int,
        dropout: float = 0.1,
    ):
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            d_model=d_model,
            dropout=dropout,
            use_channel_position=False,
        )


class SpikePredictionHead(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, prior_prob: float = 0.01):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        # 用先验正类概率初始化 bias，避免训练初期输出整体过高。
        prior_prob = float(min(max(prior_prob, 1e-5), 1.0 - 1e-5))
        bias = math.log(prior_prob / (1.0 - prior_prob))
        nn.init.constant_(self.mlp[-1].bias, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        return self.mlp(x).squeeze(-1)  # [B, L]


def build_local_attn_mask(
    L: int,
    window: int,
    causal: bool,
    device: torch.device,
    dtype: torch.dtype = torch.bool,
) -> torch.Tensor:
    i = torch.arange(L, device=device).unsqueeze(1)
    j = torch.arange(L, device=device).unsqueeze(0)

    if causal:
        disallowed = (j > i) | ((i - j) > window)
    else:
        disallowed = (i - j).abs() > window

    if dtype == torch.bool:
        return disallowed

    mask = torch.zeros((L, L), device=device, dtype=dtype)
    mask[disallowed] = float("-inf")
    return mask


def apply_rotary_embedding(
    tensor: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> torch.Tensor:
    """Apply RoPE to ``[B, H, L, D]`` queries or keys.

    Adjacent feature pairs form the two-dimensional rotation planes.  RoPE is
    applied inside attention rather than added to the token representation.
    """
    if tensor.ndim != 4 or tensor.shape[-1] % 2:
        raise ValueError(
            "RoPE expects [B,H,L,D] with an even head dimension, got "
            f"{tuple(tensor.shape)}"
        )
    even = tensor[..., 0::2]
    odd = tensor[..., 1::2]
    rotated = torch.stack(
        (even * cosine - odd * sine, even * sine + odd * cosine), dim=-1
    )
    return rotated.flatten(-2)


class RotarySelfAttention(nn.Module):
    """Multi-head self-attention with rotary query/key position encoding."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float,
        max_len: int,
        rope_base: float = 10000.0,
        qk_normalization: str = "none",
    ):
        super().__init__()
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.head_dim = self.d_model // self.nhead
        if self.head_dim % 2:
            raise ValueError("RoPE requires an even attention head dimension")
        self.dropout = float(dropout)
        if qk_normalization not in {"none", "l2"}:
            raise ValueError("qk_normalization must be 'none' or 'l2'")
        self.qk_normalization = qk_normalization
        if qk_normalization == "l2":
            # Persistent marker prevents silently restoring cosine-attention
            # weights into the historical unnormalised forward path.
            self.register_buffer(
                "qk_cosine_scale", torch.tensor(math.sqrt(self.head_dim))
            )
        self.qkv = nn.Linear(self.d_model, 3 * self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        inverse_frequency = 1.0 / (
            float(rope_base)
            ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        positions = torch.arange(max_len, dtype=torch.float32)
        phase = torch.outer(positions, inverse_frequency)
        self.register_buffer("rope_cos", phase.cos(), persistent=False)
        self.register_buffer("rope_sin", phase.sin(), persistent=False)

    def normalize_query_key(
        self, query: torch.Tensor, key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the checkpoint-compatible fixed-temperature cosine attention."""
        if self.qk_normalization == "none":
            return query, key

        def normalize(tensor: torch.Tensor) -> torch.Tensor:
            work = (
                tensor.float()
                if tensor.dtype in (torch.float16, torch.bfloat16)
                else tensor
            )
            return (
                F.normalize(work, p=2.0, dim=-1, eps=1e-6)
                * self.qk_cosine_scale
            ).to(tensor.dtype)

        return normalize(query), normalize(key)

    def forward(
        self,
        x: torch.Tensor,
        disallowed_mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch, length, _ = x.shape
        if length > self.rope_cos.shape[0]:
            raise ValueError(
                f"Sequence length {length} exceeds RoPE max_len={self.rope_cos.shape[0]}"
            )
        qkv = self.qkv(x).reshape(
            batch, length, 3, self.nhead, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        cosine = self.rope_cos[:length].to(dtype=query.dtype)[None, None]
        sine = self.rope_sin[:length].to(dtype=query.dtype)[None, None]
        query = apply_rotary_embedding(query, cosine, sine)
        key = apply_rotary_embedding(key, cosine, sine)
        query, key = self.normalize_query_key(query, key)

        allowed_mask: torch.Tensor | None = None
        if disallowed_mask is not None:
            allowed_mask = ~disallowed_mask.bool()
        if key_padding_mask is not None:
            key_allowed = ~key_padding_mask.bool()
            if allowed_mask is None:
                allowed_mask = key_allowed[:, None, None, :]
            else:
                allowed_mask = allowed_mask[None, None] & key_allowed[:, None, None, :]
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=allowed_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(batch, length, self.d_model)
        return self.out_proj(attended)


class RotaryTransformerEncoderLayer(nn.Module):
    """Pre-norm Transformer layer whose temporal attention uses RoPE."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        max_len: int,
        qk_normalization: str = "none",
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.self_attn = RotarySelfAttention(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            max_len=max_len,
            qk_normalization=qk_normalization,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        disallowed_mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        normalized = self.norm1(x)
        x = x + self.dropout1(
            self.self_attn(normalized, disallowed_mask, key_padding_mask)
        )
        normalized = self.norm2(x)
        feedforward = self.linear2(self.dropout(F.gelu(self.linear1(normalized))))
        return x + self.dropout2(feedforward)


class RotaryTransformerEncoder(nn.Module):
    """Stack of rotary pre-norm Transformer encoder layers."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        max_len: int,
        qk_normalization: str = "none",
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            RotaryTransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                max_len=max_len,
                qk_normalization=qk_normalization,
            )
            for _ in range(num_layers)
        )

    def forward(
        self,
        x: torch.Tensor,
        disallowed_mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, disallowed_mask, key_padding_mask)
        return x


class SpikeDetector(nn.Module):
    """
    新版 Spike detector：
    - 默认前端保持历史行为；可选 geometry_aware 前端使用物理 channel position
    - 使用时间卷积前端提取局部波形
    - 再用局部 Transformer 建模长程上下文

    输入:
        x: [B, L, C]
        channel_position: 保留接口兼容，但不会被使用
        time_valid_mask: [B, L]，True 表示有效时间点
    输出:
        logits: [B, L]
    """

    def __init__(
        self,
        num_channels: int = 4,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_ff: int = 512,
        dropout: float = 0.1,
        max_len: int = 50000,
        frontend_channels: int = 64,
        local_window: int = 256,
        causal: bool = False,
        prior_prob: float = 0.01,
        use_positional_encoding: bool = True,
        positional_encoding_type: str | None = None,
        frontend_type: str = "standard",
        qk_normalization: str = "none",
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead}).")

        self.num_channels = num_channels
        self.d_model = d_model
        self.local_window = local_window
        self.causal = causal
        if positional_encoding_type is None:
            positional_encoding_type = (
                "sinusoidal" if use_positional_encoding else "none"
            )
        self.positional_encoding_type = str(positional_encoding_type)
        if self.positional_encoding_type not in {"none", "sinusoidal", "rotary"}:
            raise ValueError(
                "positional_encoding_type must be 'none', 'sinusoidal', or 'rotary'"
            )
        self.use_positional_encoding = self.positional_encoding_type != "none"
        if qk_normalization not in {"none", "l2"}:
            raise ValueError("qk_normalization must be 'none' or 'l2'")
        if qk_normalization != "none" and self.positional_encoding_type != "rotary":
            raise ValueError(
                "Q/K normalization is currently implemented for rotary attention only"
            )
        self.qk_normalization = qk_normalization
        self.frontend_type = str(frontend_type)

        frontend_classes = {
            "standard": ConvFrontend,
            "standard_groupnorm": GroupNormConvFrontend,
            "multiscale": MultiscaleConvFrontend,
            "dilated_residual": DilatedResidualConvFrontend,
            "multiscale_dilated": MultiscaleDilatedResidualFrontend,
            "geometry_aware": GeometryAwareConvFrontend,
            "geometry_residual": GeometryResidualConvFrontend,
            "geometry_residual_no_position": GeometryResidualNoPositionConvFrontend,
        }
        if self.frontend_type not in frontend_classes:
            raise ValueError(
                f"Unknown frontend_type={self.frontend_type!r}; "
                f"expected one of {sorted(frontend_classes)}"
            )
        self.frontend = frontend_classes[self.frontend_type](
            in_channels=num_channels,
            hidden_channels=frontend_channels,
            d_model=d_model,
            dropout=dropout,
        )
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=max_len)
        self.input_dropout = nn.Dropout(dropout)

        self.num_layers = int(num_layers)
        if self.num_layers < 0:
            raise ValueError("num_layers must be non-negative")
        if self.num_layers == 0:
            self.transformer_encoder = nn.Identity()
        elif self.positional_encoding_type == "rotary":
            self.transformer_encoder = RotaryTransformerEncoder(
                d_model=d_model,
                nhead=nhead,
                num_layers=self.num_layers,
                dim_feedforward=dim_ff,
                dropout=dropout,
                max_len=max_len,
                qk_normalization=qk_normalization,
            )
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_ff,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
                activation="gelu",
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=self.num_layers
            )
        self.head = SpikePredictionHead(d_model=d_model, dropout=dropout, prior_prob=prior_prob)

        self._mask_cache: dict[tuple[int, int, bool, str], torch.Tensor] = {}

    def _get_attn_mask(self, L: int, device: torch.device) -> torch.Tensor:
        key = (L, self.local_window, self.causal, device.type)
        mask = self._mask_cache.get(key)
        if mask is None or mask.device != device:
            mask = build_local_attn_mask(
                L=L,
                window=self.local_window,
                causal=self.causal,
                device=device,
                dtype=torch.bool,
            )
            self._mask_cache[key] = mask
        return mask

    def forward_features(
        self,
        x: torch.Tensor,
        channel_position: Optional[torch.Tensor] = None,
        time_valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return the contextual per-sample representation ``[B, L, D]``.

        This is intentionally a side-effect-free extraction point for auxiliary
        heads.  Keeping the original dense prediction head separate lets a
        candidate classifier reuse a frozen detector without changing the
        detector's public output shape or any historical checkpoints.
        """
        if x.dim() != 3:
            raise ValueError(f"Expected x to have shape [B, L, C], got {tuple(x.shape)}")
        if x.shape[-1] != self.num_channels:
            raise ValueError(f"Expected num_channels={self.num_channels}, got input last dim={x.shape[-1]}")

        B, L, C = x.shape
        _ = B, C

        if self.frontend_type in {
            "geometry_aware",
            "geometry_residual",
            "geometry_residual_no_position",
        }:
            x = self.frontend(x, channel_position)  # [B, L, D]
        else:
            x = self.frontend(x)                 # [B, L, D]
        # Absolute position is retained by default for checkpoint/API
        # compatibility.  Detection is intrinsically translation equivariant,
        # so experiments may disable it without changing parameter shapes.
        if self.positional_encoding_type == "sinusoidal":
            x = self.positional_encoding(x)  # [B, L, D]
        x = self.input_dropout(x)

        src_key_padding_mask = None
        if time_valid_mask is not None:
            src_key_padding_mask = ~time_valid_mask.bool()

        if self.num_layers > 0:
            attn_mask = self._get_attn_mask(L, x.device)
            if self.positional_encoding_type == "rotary":
                x = self.transformer_encoder(
                    x,
                    disallowed_mask=attn_mask,
                    key_padding_mask=src_key_padding_mask,
                )
            else:
                x = self.transformer_encoder(
                    x,
                    mask=attn_mask,
                    src_key_padding_mask=src_key_padding_mask,
                )
        else:
            x = self.transformer_encoder(x)
        return x

    def forward(
        self,
        x: torch.Tensor,
        channel_position: Optional[torch.Tensor] = None,
        time_valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        features = self.forward_features(
            x,
            channel_position=channel_position,
            time_valid_mask=time_valid_mask,
        )
        return self.head(features)
