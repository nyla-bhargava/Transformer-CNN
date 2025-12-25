import torch.nn as nn

class CNNOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class TransformerOnly(nn.Module):
    def __init__(self, d_model=768, heads=8, layers=2):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(d_model, heads, dim_feedforward=1024)
        self.transformer = nn.TransformerEncoder(enc_layer, layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.transformer(x).squeeze(1)
        return self.fc(x)


class TransformerCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = TransformerOnly()
        self.cnn = CNNOnly()
        # NOTE: fuse transformer + CNN as in your original architecture

    def forward(self, x):
        t_out = self.transformer(x)
        c_out = self.cnn(x)
        combined = torch.cat([t_out, c_out], dim=-1)
        return combined
