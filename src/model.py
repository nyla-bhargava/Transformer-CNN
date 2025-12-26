import torch
import torch.nn as nn

class CNNOnly(nn.Module):
    """
    CNN baseline operating on DNABERT embeddings
    Input: (batch, seq_len, 768)
    """
    def __init__(self, embed_dim=768):
        super().__init__()

        self.conv = nn.Conv1d(embed_dim, 256, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)

        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        # x: (batch, seq_len, embed_dim)
        x = x.transpose(1, 2)              # (batch, embed_dim, seq_len)
        x = self.relu(self.conv(x))
        x = self.pool(x).squeeze(-1)       # (batch, 256)
        x = self.dropout(x)
        return self.fc(x)


class TransformerOnly(nn.Module):
    """
    Transformer baseline operating on DNABERT embeddings
    Input: (batch, seq_len, 768)
    """
    def __init__(self, embed_dim=768, heads=8, layers=2):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=1024,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, layers)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, x):
        # x: (batch, seq_len, embed_dim)
        x = self.transformer(x)             # (batch, seq_len, embed_dim)
        x = x.transpose(1, 2)               # (batch, embed_dim, seq_len)
        x = self.pool(x).squeeze(-1)        # (batch, embed_dim)
        return self.fc(x)

class TransformerCNN(nn.Module):
    """
    Hybrid Transformer–CNN model
    """
    def __init__(self, embed_dim=768):
        super().__init__()

        self.transformer = TransformerOnly(embed_dim)
        self.cnn = CNNOnly(embed_dim)

        self.fc_fusion = nn.Linear(2, 1)

    def forward(self, x):
        t_out = self.transformer(x)   # (batch, 1)
        c_out = self.cnn(x)           # (batch, 1)

        fused = torch.cat([t_out, c_out], dim=1)
        return self.fc_fusion(fused)
