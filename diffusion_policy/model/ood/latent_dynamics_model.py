"""LPB 스타일의 latent dynamics predictor를 단일 policy encoder 위에 맞춘 모델."""

import torch
import torch.nn as nn


class TemporalChunkEmbedding(nn.Module):
    """LPB의 proprio/action chunk encoder와 비슷한 1D 토큰 임베딩."""

    def __init__(self, in_chans: int, emb_dim: int):
        super().__init__()
        self.in_chans = in_chans
        self.emb_dim = emb_dim
        self.proj = nn.Conv1d(
            in_channels=in_chans,
            out_channels=emb_dim,
            kernel_size=1,
            stride=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape [B, T, D], got {tuple(x.shape)}")
        x = x.transpose(1, 2)
        x = self.proj(x)
        return x.transpose(1, 2)


class LatentDynamicsModel(nn.Module):
    """
    기존 policy encoder가 만든 visual latent를 기준으로,
    현재 latent/proprio와 미래 action chunk를 temporal transformer에 넣어
    horizon 시점의 latent/proprio를 예측하는 OOD용 dynamics 모델.
    """

    def __init__(
        self,
        latent_dim: int,
        proprio_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
        proprio_emb_dim: int = 128,
        action_emb_dim: int = 128,
        num_layers: int = 6,
        num_heads: int = 16,
        dropout: float = 0.1,
        max_action_horizon: int = 16,
    ):
        super().__init__()
        if hidden_dim <= (proprio_emb_dim + action_emb_dim):
            raise ValueError("hidden_dim must be larger than proprio_emb_dim + action_emb_dim")

        self.latent_dim = latent_dim
        self.proprio_dim = proprio_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.proprio_emb_dim = proprio_emb_dim
        self.action_emb_dim = action_emb_dim
        self.visual_emb_dim = hidden_dim - proprio_emb_dim - action_emb_dim
        self.max_action_horizon = max_action_horizon

        self.visual_encoder = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, self.visual_emb_dim),
        )
        self.proprio_encoder = TemporalChunkEmbedding(
            in_chans=proprio_dim,
            emb_dim=proprio_emb_dim,
        )
        self.action_encoder = TemporalChunkEmbedding(
            in_chans=action_dim,
            emb_dim=action_emb_dim,
        )

        self.input_norm = nn.LayerNorm(hidden_dim)
        self.pos_embed = nn.Parameter(
            torch.randn(1, max_action_horizon, hidden_dim) * 0.02
        )
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim),
        )

        self.visual_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.proprio_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, proprio_dim),
        )

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        return torch.triu(mask, diagonal=1)

    def forward(
        self,
        latent: torch.Tensor,
        proprio: torch.Tensor,
        action_seq: torch.Tensor,
    ):
        if latent.ndim != 2:
            raise ValueError(f"Expected latent shape [B, D], got {tuple(latent.shape)}")
        if proprio.ndim != 2:
            raise ValueError(f"Expected proprio shape [B, D], got {tuple(proprio.shape)}")
        if action_seq.ndim != 3:
            raise ValueError(f"Expected action_seq shape [B, H, D], got {tuple(action_seq.shape)}")
        if action_seq.shape[1] > self.max_action_horizon:
            raise ValueError(
                f"action horizon {action_seq.shape[1]} exceeds max_action_horizon {self.max_action_horizon}"
            )

        batch_size, horizon, _ = action_seq.shape

        visual_token = self.visual_encoder(latent).unsqueeze(1).expand(batch_size, horizon, -1)
        proprio_token = self.proprio_encoder(proprio.unsqueeze(1)).expand(batch_size, horizon, -1)
        action_tokens = self.action_encoder(action_seq)

        tokens = torch.cat([visual_token, proprio_token, action_tokens], dim=-1)
        tokens = self.input_norm(tokens)
        tokens = tokens + self.pos_embed[:, :horizon]
        tokens = self.dropout(tokens)

        encoded = self.transformer(tokens, mask=self._causal_mask(horizon, tokens.device))
        summary = encoded[:, -1]
        future_latent = self.visual_head(summary)
        future_proprio = self.proprio_head(summary)
        return future_latent, future_proprio
