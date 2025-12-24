import torch
import torch.nn as nn


class Stage2Model(nn.Module):
    """
    Transformer–CNN hybrid model for CRISPR off-target prediction
    """

    def __init__(self, sg_dim: int):
        super().__init__()

        self.input_proj = nn.Linear(10, 128)

        self.cnn = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.sg_proj = nn.Linear(sg_dim, 128)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(256, 1)

    def forward(self, pair, mv, pam, sg_emb):
        """
        pair : (B, L, 8)
        mv   : (B, L)
        pam  : (B, L)
        sg_emb : (B, sg_dim)
        """
        x = torch.cat(
            [pair, mv.unsqueeze(-1), pam.unsqueeze(-1)],
            dim=-1
        )  # (B, L, 10)

        x = self.input_proj(x)
        x = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        x = self.transformer(x)

        pooled = x.mean(dim=1)
        sg_feat = self.sg_proj(sg_emb)

        fused = torch.cat([pooled, sg_feat], dim=1)
        fused = self.dropout(fused)

        return self.classifier(fused).squeeze(-1)
