"""HGAM: LSTM + hypergraph attention (Sun et al.). Returns h_{t,m} as [B, d, 1, M]."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attn_head_adj(nn.Module):
    def __init__(self,
                 in_channel,
                 out_sz, 
                 in_drop=0.0,
                 coef_drop=0.0,
                 activation=nn.ELU(),
                 residual=True):
        super(Attn_head_adj, self).__init__()

        self.in_channel = in_channel
        self.out_sz = out_sz
        self.in_drop = in_drop
        self.coef_drop = coef_drop
        self.activation = activation
        self.residual = residual

        self.conv1 = nn.Conv2d(self.in_channel, self.out_sz, (1, 1))
        self.conv2_1 = nn.Conv2d(self.out_sz, 1, (1, 1))
        self.conv2_2 = nn.Conv2d(self.out_sz, 1, (1, 1))
        self.leakyrelu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)
        self.in_dropout = nn.Dropout()
        self.coef_dropout = nn.Dropout()
        if self.in_channel != self.out_sz:
            self.res_conv = nn.Conv2d(self.in_channel, self.out_sz, (1, 1))

    def forward(self, x, mask): # B, C, T, N
        seq = x
        B, _, T, N = seq.size()

        seq_fts = self.conv1(seq)   # B, C, T, N
        f_1 = self.conv2_1(seq_fts).squeeze(1)  # B, T, N
        f_2 = self.conv2_2(seq_fts).squeeze(1)  # B, T, N

        logits = f_1[:, :, None, :] + f_2[:, :, :, None]
        logits = self.leakyrelu(logits)
        attention = logits  # B, T, N, N

        mask = mask[:, None, :, None].int() + mask[:, None, None, :].int()  # B, 1, N, N
        mask = torch.where(mask > 0, torch.zeros_like(mask), -1e8)
        attention = (attention + mask).contiguous() # B, T, N, N
        attention = attention.view(-1, N, N)    # B * T, N, N

        coefs = self.softmax(attention)

        if self.coef_drop != 0.0:
            coefs = self.coef_dropout(coefs)

        seq_fts = seq_fts.permute(0, 2, 3, 1).contiguous().view(-1, N, self.out_sz)
        if self.in_dropout != 0.0:
            seq_fts = self.in_dropout(seq_fts)

        ret = torch.matmul(coefs, seq_fts)  # B * T, N, C
        ret = ret.view(-1, T, N, self.out_sz).permute(0, 3, 1, 2).contiguous()
        # ret = torch.transpose(seq_fts, 2, 1).contiguous()

        if self.residual:
            if seq.shape[1] != ret.shape[1]:
                ret = ret + self.res_conv(seq)
            else:
                ret = ret + seq
        
        return self.activation(ret)  # B, C, T, N
        
class HGAT(nn.Module):
    def __init__(self, nfeat, nhid,  dropout):
        ''' attention mechanism. '''
        super(HGAT, self).__init__()
        self.dropout = dropout

        self.attention = Attn_head_adj(nfeat, nhid, in_drop=0.6, coef_drop=0.6, activation=nn.ELU(),
                                         residual=True)

    def forward(self, x, mask):
        x = F.dropout(x, self.dropout, training=self.training)

        x = self.attention(x, mask)
        x = F.dropout(x, self.dropout, training=self.training)

        return x
    
class HGAMBackbone(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(self, in_channels=4, d=16, num_layers=2):
        super().__init__()

        dropout = 0.5

        self.rnn = nn.LSTM(in_channels,
                          d * 2,
                          num_layers=num_layers,
                          batch_first=True,
                          bidirectional=False)

        self.HGAT = HGAT(nfeat=d * 2,
                         nhid=d,
                         dropout=dropout,
                         )

    def forward(self, price_series, mask):  # [B, C, T, N]
        src_seq = price_series.permute(0, 3, 2, 1).contiguous()
        batch = src_seq.size(0)
        stock = src_seq.size(1)
        seq_len = src_seq.size(2)
        dim = src_seq.size(3)
        src_seq = torch.reshape(src_seq, (batch*stock, seq_len, dim))

        rnn_output, *_ = self.rnn(src_seq)

        rnn_output = rnn_output[:, -1, :]
        enc_output = torch.reshape(rnn_output, (batch, stock, -1))  # B, N, C

        enc_output = enc_output.permute(0, 2, 1).contiguous()[:, :, None, :]  # B, C, 1, N

        HGAT_output = self.HGAT(enc_output, mask)

        return HGAT_output
