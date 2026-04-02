import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Resize
from models.v3_model.baseline import R3DBackbone
from einops import repeat
import cv2
import numpy as np
import os


def generate_orthogonal_vectors(N, d):
    assert N >= d, "[generate_orthogonal_vectors] dim issue"
    init_vectors = torch.normal(0, 1, size=(N, d))
    norm_vectors = nn.functional.normalize(init_vectors, p=2.0, dim=1)
    q, r = torch.linalg.qr(norm_vectors)
    d = torch.diag(r, 0)
    ph = d.sign()
    q *= ph
    return q

def get_orthogonal_queries(n_classes, n_dim, apply_norm=True):
    if n_classes < n_dim:
        vecs = generate_orthogonal_vectors(n_dim, n_dim)[:n_classes, :]
    else:
        vecs = generate_orthogonal_vectors(n_dim, n_dim)
    if apply_norm:
        vecs = nn.functional.normalize(vecs, p=2.0, dim=1)
    return vecs

class SharedDecoder(nn.Module):
    def __init__(self, d_model=512, num_heads=8, num_layers=2, num_queries=30):
        super().__init__()
        self.positional_encoding = nn.Parameter(torch.zeros(16, d_model), requires_grad=True)
        decoder_layer = nn.TransformerDecoderLayer(d_model, num_heads, dim_feedforward=256, activation='gelu', batch_first=True, dropout=0.0, norm_first=False)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.action_tokens = nn.Parameter(torch.unsqueeze(get_orthogonal_queries(num_queries, d_model), dim=0))
        self.view_tokens = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, x):
        b, t, d = x.shape
        x = x + self.positional_encoding[:t]
        action_tokens = repeat(self.action_tokens, '() n d -> b n d', b=b)
        view_tokens = repeat(self.view_tokens, '() n d -> b n d', b=b)
        queries = torch.cat((action_tokens, view_tokens), dim=1)
        out = self.decoder(queries, x)
        x_act, x_view = out[:, :-1, :], out[:, -1, :]
        features_view = x_view
        features_action = x_act.mean(dim=1)
        return features_action, features_view, action_tokens, view_tokens


class CNN2D_Transformer(nn.Module):
    def __init__(self, video_encoder, len_feature, num_classes, num_frames, fusion_type, config=None):
        super(CNN2D_Transformer, self).__init__()
        self.fusion_type = fusion_type
        self.motion_score = config.motion_score
        self.video_encoder = R3DBackbone()  # pre-defined encoder
        self.decoder = SharedDecoder(d_model=512, num_queries=30)  # shared across all views

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512, nhead=4, dim_feedforward=256, dropout=0.1, batch_first=True
        )
        self.device_transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)


        if fusion_type == 'concat':
            self.mlp_head = nn.Linear(512 * 4, num_classes)
        else:
            self.mlp_head = nn.Linear(512, num_classes)

        self.mlp_head_view = nn.Linear(512, config.num_views)  # view prediction head

    def compute_motion_weights(self, x):
        # x: (B, V, T, C, H, W)
        diffs = torch.abs(x[:, :, 3:] - x[:, :, :-3])  # (B, V, T-3, C, H, W)
        motion_score_bitmask = diffs > 0.1
        motion_score_bit_sum = motion_score_bitmask.sum(dim=[2, 3, 4, 5])  # (B, V)
        weights_v3 = motion_score_bit_sum / motion_score_bit_sum.sum(dim=1, keepdim=True)
        return weights_v3

    def forward(self, x, is_training):
        # x: [B, V, T, C, H, W]
        B, V, T, C, H, W = x.shape

        if self.motion_score:
            motion_weights = self.compute_motion_weights(x)

        x = x.view(B * V, T, C, H, W)  # [B*V, C, T, H, W]
        features = self.video_encoder(x)  # [B*V, D, T', 1, 1]
        features = features.squeeze(-1).squeeze(-1).permute(0, 2, 1)  # [B*V, T', D]
        features_action, features_view, action_tokens, view_tokens = self.decoder(features)  # [B*V, D], [B*V, D], [B*V, 1, D], [B*V, 30, D]

        features_action = features_action.view(B, V, -1)  # [B, V, D]
        features_view = features_view.view(B, V, -1)  # [B, V, D]

        pred_views = self.mlp_head_view(features_view)  # [B, V, num_views]

        if self.motion_score and motion_weights is not None:
            features_action = features_action * motion_weights.unsqueeze(-1)  # [B, V, 1]

        if self.fusion_type == 'transformer':
            x_device = self.device_transformer(features_action)
            fused = x_device.sum(dim=1)
        elif self.fusion_type == 'max':
            fused = features_action.max(dim=1)[0]
        elif self.fusion_type == 'mean':
            fused = torch.mean(features_action, dim=1)
        elif self.fusion_type == 'sum':
            fused = torch.sum(features_action, dim=1)
        elif self.fusion_type == 'concat':
            fused = features_action.reshape(B, -1)
        else:
            raise ValueError("Invalid fusion_type")

        # reshape action_tokens to [B*V*30, D]
        action_tokens = action_tokens.view(B * V, -1, action_tokens.shape[-1])

        out = self.mlp_head(fused)
        # [TESTED] added features_action to return
        return out, pred_views, fused, action_tokens, view_tokens, features_action
