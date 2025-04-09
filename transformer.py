import torch
import torch.nn as nn
import torch.nn.functional as F

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=5000, learnable=False):
        super().__init__()
        self.dim = dim
        self.learnable = learnable
        
        if self.learnable:
            self.pos_embedding = nn.Embedding(max_len, dim)
        else:
            # Sinusoidal positional encoding
            pe = torch.zeros(max_len, dim)
            position = torch.arange(0, max_len).unsqueeze(1).float()
            div_term = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe)
    
    def forward(self, x):
        if self.learnable:
            return x + self.pos_embedding(torch.arange(x.size(1), device=x.device))
        else:
            return x + self.pe[:x.size(1)].unsqueeze(0)


# Patch Embedding Layer (1D signal into patches)
class PatchEmbedding(nn.Module):
    def __init__(self, input_dim, patch_size, stride):
        super().__init__()
        self.conv = nn.Conv1d(input_dim, input_dim, kernel_size=patch_size, stride=stride)
    
    def forward(self, x):
        return self.conv(x)  # Outputs patches

# CNN for Local Feature Extraction
class CNN1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., patch_size=16, stride=8, learnable_pos_enc=False):
        super().__init__()
        self.patch_embed = PatchEmbedding(dim, patch_size, stride)
        self.positional_encoding = PositionalEncoding(dim, learnable=learnable_pos_enc)
        self.cnn = CNN1D(in_channels=dim, out_channels=dim)
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x):
        x = self.cnn(x)  # CNN for local feature extraction
        x = self.patch_embed(x)  # Patch Embedding (Conv1D)
        x = self.positional_encoding(x)  # Add positional encoding
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class EfficientAttention(nn.Module):
    """Memory-efficient multi-head attention"""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)

class Mlp(nn.Module):
    """MLP from Vision Transformer with added dropout"""
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# ECG Transformer Model
import torch
import torch.nn as nn
import torch.nn.functional as F

class ECGTransformer(nn.Module):
    def __init__(self, input_channels=12, seq_len=1024, patch_size=50,
                 embed_dim=48, num_heads=8, num_layers=6, num_classes=5,
                 expansion=4, dropout=0.1):
        super(ECGTransformer, self).__init__()

        # Ensure input_channels is 12 (ECG leads) and seq_len is 4096
        self.input_channels = input_channels
        self.seq_len = seq_len

        # CNN Stem to reduce dimensionality
        self.cnn_stem = nn.Sequential(
            nn.Conv1d(input_channels, embed_dim, kernel_size=15, stride=2, padding=7),  # input_channels=12
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2),  # Reduce size further
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Transformer Encoder Layers
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True),
            num_layers=num_layers
        )

        # Fully connected layer for classification
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # Input shape: (batch_size, input_channels, seq_len)
        # Pass through CNN Stem (to reduce dimensions)
        x = self.cnn_stem(x)

        # After the CNN stem, x shape will be (batch_size, embed_dim, reduced_seq_len)
        # Transpose it to (seq_len, batch_size, embed_dim) for the Transformer
        x = x.permute(2, 0, 1)   # Now shape (seq_len, batch_size, embed_dim)

        # Pass through Transformer Encoder
        x = self.transformer_encoder(x)

        # Use the output from the final timestep of the Transformer
        x = x.mean(dim=0)  # Get the last timestep output

        # Fully connected layer for classification
        x = self.fc(x)
        return x


# Example Usage
if __name__ == "__main__":
    model = ECGTransformer(input_channels=12, seq_len=5000, patch_size=50,
                           embed_dim=128, num_heads=8, num_layers=6,
                           num_classes=5, dropout=0.1)

    sample_input = torch.randn(16, 12, 5000)  # (Batch=16, Channels=12, Seq Len=5000)
    output = model(sample_input)
    print(output.shape)  # Expected Output: (16, 5)
