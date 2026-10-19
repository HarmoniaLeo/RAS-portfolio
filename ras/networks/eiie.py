"""Feature extractors of the original EIIE framework (Jiang et al., 2017).

CNN EIIE uses a temporal convolution (TCN) and bRNN EIIE an LSTM over the price
pattern. Both return the embedding h_{t,m} as a tensor of shape [B, d, 1, M].
"""
import torch.nn as nn


class EIIEConv(nn.Module):
    """CNN EIIE."""

    def __init__(self, in_channels=4, d=20, num_layers=1, window=50, out_channels=2, kernel_size=3):
        super().__init__()
        time_horizon = window - 1
        self.convLayers = nn.Sequential()
        self.convLayers.append(nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1)))
        self.convLayers.append(nn.ReLU())
        for i in range(1, num_layers):
            self.convLayers.append(nn.Conv2d(out_channels * 2 ** (i - 1), out_channels * 2 ** i, kernel_size=(kernel_size, 1)))
            self.convLayers.append(nn.ReLU())
        self.conv2 = nn.Conv2d(out_channels * 2 ** (num_layers - 1), d,
                               kernel_size=(time_horizon - num_layers * kernel_size + num_layers, 1), stride=(time_horizon, 1))
        self.relu2 = nn.ReLU()

    def forward(self, patterns, asset_mask=None):  # [B, C, l, M]
        return self.relu2(self.conv2(self.convLayers(patterns)))


class EIIELSTM(nn.Module):
    """bRNN EIIE."""

    def __init__(self, in_channels=4, d=20, num_layers=1):
        super().__init__()
        self.rnn = nn.LSTM(in_channels, d, num_layers, batch_first=False)
        self.d = d

    def forward(self, patterns, asset_mask=None):  # [B, C, l, M]
        b, c, length, m = patterns.shape
        x = patterns.permute(2, 0, 3, 1).contiguous().view(length, b * m, c)
        x, _ = self.rnn(x)
        return x[-1:].view(b, m, 1, self.d).permute(0, 3, 2, 1).contiguous()
