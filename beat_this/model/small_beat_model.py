# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# # ResidualBlock remains unchanged
# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, num_filters, kernel_size, dilation_rate, dropout_rate=0.0):
#         super(ResidualBlock, self).__init__()
#         self.conv1 = nn.Conv1d(
#             in_channels,
#             num_filters,
#             kernel_size,
#             padding=dilation_rate * (kernel_size - 1) // 2,
#             dilation=dilation_rate,
#         )
#         self.conv2 = nn.Conv1d(
#             num_filters,
#             num_filters,
#             kernel_size,
#             padding=dilation_rate * (kernel_size - 1) // 2,
#             dilation=dilation_rate,
#         )
#         self.conv_residual = nn.Conv1d(in_channels, num_filters, kernel_size=1)
#         self.dropout = nn.Dropout(dropout_rate)
#         self.activation = nn.ELU()
#
#     def forward(self, x):
#         res_x = self.conv_residual(x)  # Residual connection
#         x = self.conv1(x)
#         x = self.activation(x)
#         x = self.dropout(x)
#         x = self.conv2(x)
#         x = self.activation(x)
#         x = self.dropout(x)
#         return res_x + x  # Add residual connection
#
# # TCN remains unchanged
# class TCN(nn.Module):
#     def __init__(self, num_filters, kernel_size, dilations, dropout_rate=0.15):
#         super(TCN, self).__init__()
#         self.residual_blocks = nn.ModuleList(
#             [ResidualBlock(num_filters, num_filters, kernel_size, d, dropout_rate) for d in dilations]
#         )
#         self.activation = nn.ELU()
#
#     def forward(self, x):
#         for block in self.residual_blocks:
#             x = block(x)
#         return self.activation(x)
#
# class BeatThisSmall(nn.Module):
#     def __init__(self, num_filters=20, kernel_size=5, num_dilations=10, dropout_rate=0.15):
#         super(BeatThisSmall, self).__init__()  # Fixed incorrect class reference
#         self.conv1 = nn.Conv2d(1, num_filters, (3, 3), padding=(1, 0))
#         self.pool1 = nn.MaxPool2d((1, 3))
#         self.conv2 = nn.Conv2d(num_filters, num_filters, (1, 20), padding=0)
#         self.pool2 = nn.MaxPool2d((1, 3))
#         self.conv3 = nn.Conv2d(num_filters, num_filters, (3, 3), padding=(1, 0))
#         self.pool3 = nn.MaxPool2d((1, 3))
#
#         self.dropout = nn.Dropout(dropout_rate)
#
#         # Temporal Convolutional Network
#         dilations = [2 ** i for i in range(num_dilations)]
#         self.tcn = TCN(num_filters, kernel_size, dilations, dropout_rate)
#
#         self.adaptive_pool = nn.AdaptiveAvgPool1d(1500)  # Output sequence length of 1500
#         self.beats_dense = nn.Conv1d(num_filters, 1, kernel_size=1)
#         self.downbeats_dense = nn.Conv1d(num_filters, 1, kernel_size=1)
#
#     def forward(self, x):
#         print(f"[BeatThisSmall] Initial input shape: {x.shape}")
#         x = x.to(dtype=self.conv1.weight.dtype)  # Match input precision with model weights
#         print(f"[BeatThisSmall] Input type: {x.dtype}, Weight type: {self.conv1.weight.dtype}")
#
#         # Add channel dimension if missing
#         if x.ndim == 3:
#             x = x.unsqueeze(1)
#         print(f"[BeatThisSmall] Shape after adding channel dimension: {x.shape}")
#
#         # Convolution and pooling
#         x = self.pool1(F.elu(self.conv1(x)))
#         print(f"[BeatThisSmall] Shape after conv1 and pool1: {x.shape}")
#         x = self.pool2(F.elu(self.conv2(x)))
#         print(f"[BeatThisSmall] Shape after conv2 and pool2: {x.shape}")
#         x = self.pool3(F.elu(self.conv3(x)))
#         print(f"[BeatThisSmall] Shape after conv3 and pool3: {x.shape}")
#
#         # Reduce frequency dimension to 1
#         x = x.squeeze(-1)
#         print(f"[BeatThisSmall] Shape after squeezing frequency dimension: {x.shape}")
#
#         # TCN
#         x = self.tcn(x)
#         print(f"[BeatThisSmall] Shape after TCN: {x.shape}")
#
#         # Adaptive pooling
#         x = self.adaptive_pool(x)
#         print(f"[BeatThisSmall] Shape after adaptive pooling: {x.shape}")
#
#         # Output layers
#         beats = self.beats_dense(x).squeeze(1)
#         downbeats = self.downbeats_dense(x).squeeze(1)
#         print(f"[BeatThisSmall] Beats output shape: {beats.shape}")
#         print(f"[BeatThisSmall] Downbeats output shape: {downbeats.shape}")
#
#         # Return a dictionary instead of a tuple
#         return {"beat": beats, "downbeat": downbeats}



import torch
import torch.nn as nn
import torch.nn.functional as F


# ResidualBlock
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
        print(f"[ResidualBlock] Input shape: {x.shape}")
        res_x = self.conv_residual(x)  # Residual connection
        print(f"[ResidualBlock] Residual shape after conv_residual: {res_x.shape}")
        x = self.conv1(x)
        x = self.activation(x)
        x = self.dropout(x)
        print(f"[ResidualBlock] Shape after conv1: {x.shape}")
        x = self.conv2(x)
        x = self.activation(x)
        x = self.dropout(x)
        print(f"[ResidualBlock] Shape after conv2: {x.shape}")
        return res_x + x  # Add residual connection


# TCN
class TCN(nn.Module):
    def __init__(self, num_filters, kernel_size, dilations, dropout_rate=0.15):
        super(TCN, self).__init__()
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(num_filters, num_filters, kernel_size, d, dropout_rate) for d in dilations]
        )
        self.activation = nn.ELU()

    def forward(self, x):
        print(f"[TCN] Input shape: {x.shape}")
        for i, block in enumerate(self.residual_blocks):
            x = block(x)
            print(f"[TCN] Shape after ResidualBlock {i}: {x.shape}")
        return self.activation(x)


# BeatThisSmall
class BeatThisSmall(nn.Module):
    def __init__(self, num_filters=20, kernel_size=5, num_dilations=10, dropout_rate=0.15):
        super(BeatThisSmall, self).__init__()
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
        print(f"[BeatThisSmall] Initial input shape: {x.shape}")
        x = x.float()

        # Add channel dimension if missing
        if x.ndim == 3:
            x = x.unsqueeze(1)  # Shape: [batch_size, 1, time_steps, freq_bins]
        print(f"[BeatThisSmall] Shape after adding channel dimension: {x.shape}")

        # Convolution and pooling
        x = self.pool1(F.elu(self.conv1(x)))
        print(f"[BeatThisSmall] Shape after conv1 and pool1: {x.shape}")
        x = self.pool2(F.elu(self.conv2(x)))
        print(f"[BeatThisSmall] Shape after conv2 and pool2: {x.shape}")
        x = self.pool3(F.elu(self.conv3(x)))
        print(f"[BeatThisSmall] Shape after conv3 and pool3: {x.shape}")

        # # Reduce frequency dimension to 1
        x = x.squeeze(-1)

        # TCN expects [batch_size, num_filters, time_steps]
        x = self.tcn(x)  # Pass through Temporal Convolutional Network
        print(f"[BeatThisSmall] Shape after TCN: {x.shape}")

        # Downsample with adaptive pooling
        x = self.adaptive_pool(x)  # Shape: [batch_size, num_filters, 1500]
        print(f"[BeatThisSmall] Shape after adaptive pooling: {x.shape}")

        # Output dense layers
        beats = (self.beats_dense(x)).squeeze(1)  # Shape: [batch_size, 1500]
        downbeats = (self.downbeats_dense(x)).squeeze(1)  # Shape: [batch_size, 1500]
        print(f"[BeatThisSmall] Beats output shape: {beats.shape}")
        print(f"[BeatThisSmall] Downbeats output shape: {downbeats.shape}")

        return {"beat": beats, "downbeat": downbeats}


import multiprocessing
num_workers = multiprocessing.cpu_count()
print(f"Number of CPU cores available: {num_workers}")
