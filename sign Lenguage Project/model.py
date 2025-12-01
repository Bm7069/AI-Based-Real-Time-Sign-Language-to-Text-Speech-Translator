# model.py
import torch
import torch.nn as nn

class TemporalTransformerClassifier(nn.Module):
    def __init__(self, input_dim=63, d_model=128, nhead=4, num_layers=3, num_classes=10, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, dim_feedforward=d_model*4, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.randn(1,1,d_model))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        # x: (B, T, input_dim)
        B,T,_ = x.shape
        x = self.input_proj(x)  # (B,T,d_model)
        # add cls token by concatenation then transpose for transformer (seq_len, batch, d_model)
        cls_tok = self.cls_token.expand(B, -1, -1)  # (B,1,d)
        x = torch.cat([cls_tok, x], dim=1)  # (B, T+1, d)
        x = x.permute(1,0,2)  # (T+1, B, d)
        out = self.transformer(x)  # (T+1, B, d)
        cls_out = out[0]  # (B, d)
        logits = self.fc(cls_out)
        return logits
