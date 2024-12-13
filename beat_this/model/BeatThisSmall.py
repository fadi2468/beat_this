import torch
import torch.nn as nn
import torch.nn.functional as F

# ResidualBlock remains unchanged
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, num_filters, kernel_size, dilation_rate, dropout_rate=0.0):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            num_filters,
            kernel_size,
            padding=dilation_rate * (kernel_size - 1) // 2,
            dilation=dilation_rate,
        )
        self.conv2 = nn.Conv1d(
            num_filters,
            num_filters,
            kernel_size,
            padding=dilation_rate * (kernel_size - 1) // 2,
            dilation=dilation_rate,
        )
        self.conv_residual = nn.Conv1d(in_channels, num_filters, kernel_size=1)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.ELU()

    def forward(self, x):
        res_x = self.conv_residual(x)  # Residual connection
        x = self.conv1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.dropout(x)
        return res_x + x  # Add residual connection

# TCN remains unchanged
class TCN(nn.Module):
    def __init__(self, num_filters, kernel_size, dilations, dropout_rate=0.15):
        super(TCN, self).__init__()
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(num_filters, num_filters, kernel_size, d, dropout_rate) for d in dilations]
        )
        self.activation = nn.ELU()

    def forward(self, x):
        for block in self.residual_blocks:
            x = block(x)
        return self.activation(x)

class BeatThisSmall(nn.Module):
    def __init__(self, num_filters=20, kernel_size=5, num_dilations=10, dropout_rate=0.15):
        super(BeatThisModel, self).__init__()
        self.conv1 = nn.Conv2d(1, num_filters, (3, 3), padding=(1, 0))
        self.pool1 = nn.MaxPool2d((1, 3))
        self.conv2 = nn.Conv2d(num_filters, num_filters, (1, 20), padding=0)
        self.pool2 = nn.MaxPool2d((1, 3))
        self.conv3 = nn.Conv2d(num_filters, num_filters, (3, 3), padding=(1, 0))
        self.pool3 = nn.MaxPool2d((1, 3))

        self.dropout = nn.Dropout(dropout_rate)

        # Temporal Convolutional Network
        dilations = [2 ** i for i in range(num_dilations)]
        self.tcn = TCN(num_filters, kernel_size, dilations, dropout_rate)

        self.adaptive_pool = nn.AdaptiveAvgPool1d(1500)  # Output sequence length of 1500
        self.beats_dense = nn.Conv1d(num_filters, 1, kernel_size=1)
        self.downbeats_dense = nn.Conv1d(num_filters, 1, kernel_size=1)

    def forward(self, x):
        print(f"[BeatThisModel] Initial input shape: {x.shape}")
        x = x.float()

        # Add channel dimension if missing
        if x.ndim == 3:
            x = x.unsqueeze(1)  # Shape: [batch_size, 1, time_steps, freq_bins]
        print(f"[BeatThisModel] Shape after adding channel dimension: {x.shape}")

        # Convolution and pooling
        x = self.pool1(F.elu(self.conv1(x)))
        print(f"[BeatThisModel] Shape after conv1 and pool1: {x.shape}")
        x = self.pool2(F.elu(self.conv2(x)))
        print(f"[BeatThisModel] Shape after conv2 and pool2: {x.shape}")
        x = self.pool3(F.elu(self.conv3(x)))
        print(f"[BeatThisModel] Shape after conv3 and pool3: {x.shape}")

        # Reduce frequency dimension to 1
        x = x.squeeze(-1)  # Remove frequency dimension

        # TCN expects [batch_size, num_filters, time_steps]
        x = self.tcn(x)  # Pass through Temporal Convolutional Network
        print(f"[BeatThisModel] Shape after TCN: {x.shape}")

        # Downsample with adaptive pooling
        x = self.adaptive_pool(x)  # Shape: [batch_size, num_filters, 1500]
        print(f"[BeatThisModel] Shape after adaptive pooling: {x.shape}")

        # Output dense layers
        beats = torch.sigmoid(self.beats_dense(x)).squeeze(1)  # Shape: [batch_size, 1500]
        downbeats = torch.sigmoid(self.downbeats_dense(x)).squeeze(1)  # Shape: [batch_size, 1500]
        print(f"[BeatThisModel] Beats output shape: {beats.shape}")
        print(f"[BeatThisModel] Downbeats output shape: {downbeats.shape}")

        return beats, downbeats