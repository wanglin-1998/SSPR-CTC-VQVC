import torch
import torch.nn as nn
import torch.nn.functional as F 


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)
        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)

class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)
        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)
        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal

class DynamicLSTM(nn.Module):
    def __init__(self, *args, **kwargs):
        super(DynamicLSTM, self).__init__()
        self.lstm = nn.LSTM(*args, **kwargs)

    def forward(self, x, seq_lengths=None, flatten_params=False):
        if flatten_params:
            self.lstm.flatten_parameters()
        padded_len = x.shape[1]
        if seq_lengths is not None:
            x = torch.nn.utils.rnn.pack_padded_sequence(x, seq_lengths,
                batch_first=self.lstm.batch_first)
        output, final_state = self.lstm(x)
        if seq_lengths is not None:
            output, _ = torch.nn.utils.rnn.pad_packed_sequence(output,
                batch_first=self.lstm.batch_first,
                total_length=padded_len)
        return output, final_state


class DynamicGRU(nn.Module):
    def __init__(self, *args, **kwargs):
        super(DynamicGRU, self).__init__()
        self.gru = nn.GRU(*args, **kwargs)

    def forward(self, x, seq_lengths=None, flatten_params=False):
        if flatten_params:
            self.gru.flatten_parameters()
        padded_len = x.shape[1]
        if seq_lengths is not None:
            x = torch.nn.utils.rnn.pack_padded_sequence(x, seq_lengths,
                batch_first=self.gru.batch_first)
        output, final_state = self.gru(x)
        if seq_lengths is not None:
            output, _ = torch.nn.utils.rnn.pad_packed_sequence(output,
                batch_first=self.gru.batch_first,
                total_length=padded_len)
        return output, final_state


def mask_tensor(x, mask=None):
    if mask is None:
        return x
    else:
        return x * mask