import torch as th
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from mamba_ssm import Mamba

TEN = th.Tensor

def layer_init_with_orthogonal(layer, std=1.0, bias_const=1e-6):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)


class NnSeqBnMLP(nn.Module):
    """
    MLP module with input BatchNorm, layer activation, and LayerNorm support.
    """
    def __init__(self, dims, if_inp_norm=False, if_layer_norm=True, activation=None):
        super(NnSeqBnMLP, self).__init__()
        mlp_list = []
        if if_inp_norm:
            mlp_list.append(nn.BatchNorm1d(dims[0], momentum=0.9))
        
        mlp_list.append(nn.Linear(dims[0], dims[1]))
        for i in range(1, len(dims) - 1):
            mlp_list.append(nn.GELU())
            if if_layer_norm:
                mlp_list.append(nn.LayerNorm(dims[i]))
            mlp_list.append(nn.Linear(dims[i], dims[i + 1]))
        
        if activation is not None:
            mlp_list.append(activation)
        
        self.mlp = nn.Sequential(*mlp_list)
        
        # 对最后一层 Linear 层做正交初始化
        if activation is not None:
            layer_init_with_orthogonal(self.mlp[-2], std=0.1)
        else:
            layer_init_with_orthogonal(self.mlp[-1], std=0.1)

    def forward(self, seq):
        d0, d1, _ = seq.shape
        inp = seq.reshape(d0 * d1, -1)
        out = self.mlp(inp)
        return out.reshape(d0, d1, -1)

    def reset_parameters(self, std=1.0, bias_const=1e-6):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, bias_const)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, std)
                nn.init.constant_(module.bias, 0)


class Chomp1d(nn.Module):
    """
    Trim excess padding from convolution output to maintain causal convolution alignment
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    Single temporal convolution block with two conv layers, activation, dropout and residual connection
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                                     stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.utils.weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                                     stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.GELU()
        self.init_weights()
    
    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
    
    def forward(self, x):
        # x: [batch, channels, seq_len]
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """
    Stack multiple TemporalBlocks to form TCN
    """
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                          dilation=dilation_size,
                                          padding=(kernel_size - 1) * dilation_size,
                                          dropout=dropout))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        # x: [seq_len, batch, channels] -> convert to [batch, channels, seq_len]
        x = x.permute(1, 2, 0)
        x = self.network(x)
        # convert back to [seq_len, batch, channels]
        return x.permute(2, 0, 1)


class SignalSmoother(nn.Module):
    """
    Signal smoothing module using 1D convolution for simple moving average smoothing.
    kernel_size defines the smoothing window size
    """
    def __init__(self, kernel_size=5):
        super(SignalSmoother, self).__init__()
        self.kernel_size = kernel_size
        # 使用固定均值卷积核，padding 保持序列长度不变
        kernel = th.ones(1, 1, kernel_size) / kernel_size
        self.register_buffer("kernel", kernel)
    
    def forward(self, signal):
        # signal: [seq_len, batch, feature] -> smooth each batch and feature separately
        # first reshape signal to [batch*feature, 1, seq_len]
        seq_len, batch, feat = signal.shape
        signal = signal.permute(1, 2, 0).reshape(batch * feat, 1, seq_len)
        # padding: pad both ends with same values to maintain output length
        pad = (self.kernel_size - 1) // 2
        smoothed = F.conv1d(signal, self.kernel, padding=pad)
        smoothed = smoothed.reshape(batch, feat, seq_len).permute(2, 0, 1)
        return smoothed


class MambaTCNRegNet(nn.Module):
    """
    Combines MLP, TCN, Mamba modules, attention mechanism and signal smoothing for generating trading signals
    """
    def __init__(self, inp_dim, mid_dim, out_dim, num_layers, use_signal_smoothing=True, smoothing_kernel=5,
                 use_leaky_relu=False):
        super(MambaTCNRegNet, self).__init__()
        expanded_dim = mid_dim * 2  # adjust intermediate dimension
        
        # Input mapping layer (add input normalization for stability)
        self.mlp_inp = NnSeqBnMLP(
            dims=(inp_dim, mid_dim, expanded_dim),
            if_inp_norm=True,
            if_layer_norm=True,
            activation=nn.GELU()
        )
        
        # TCN layer
        tcn_channels = [expanded_dim, expanded_dim * 2, expanded_dim * 2, expanded_dim]
        self.tcn = TemporalConvNet(expanded_dim, tcn_channels, kernel_size=5, dropout=0.2)
        
        # Four parallel Mamba modules
        mamba_dim = expanded_dim // 4
        self.mamba_modules = nn.ModuleList([
            Mamba(
                d_model=mamba_dim,
                d_state=128,
                d_conv=4,
                expand=2,
                dt_min=0.001,
                dt_max=0.1,
                dt_init="random"
            ) for _ in range(4)
        ])
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=expanded_dim,
            num_heads=8,
            dropout=0.2,
            batch_first=False
        )
        
        # Feature fusion (gating mechanism)
        self.gate = nn.Sequential(
            nn.Linear(expanded_dim * 2, expanded_dim * 2),
            nn.LayerNorm(expanded_dim * 2),
            nn.Sigmoid()
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(expanded_dim * 3, expanded_dim * 2),
            nn.LayerNorm(expanded_dim * 2),
            nn.Dropout(0.25),
            nn.GELU()
        )
        
        # Output mapping layer
        # If you think Tanh limits signal amplitude, set use_leaky_relu=True to use LeakyReLU
        if use_leaky_relu:
            output_activation = nn.LeakyReLU(0.1)
        else:
            output_activation = nn.Tanh()
        self.mlp_out = NnSeqBnMLP(
            dims=(expanded_dim * 2, expanded_dim, mid_dim, out_dim),
            if_layer_norm=True,
            activation=output_activation
        )
        
        # Residual connection
        self.residual_proj = nn.Linear(expanded_dim, expanded_dim * 2)
        
        # Signal smoothing module (optional)
        self.use_signal_smoothing = use_signal_smoothing
        if self.use_signal_smoothing:
            self.signal_smoother = SignalSmoother(kernel_size=smoothing_kernel)
        
        self._init_parameters()
    
    def forward(self, inp, hid=None):
        # Input mapping
        x = self.mlp_inp(inp)
        # TCN processing
        tcn_out = self.tcn(x)
        # Mamba module processing: split x into 4 chunks and process through Mamba modules
        x_chunks = th.chunk(x, 4, dim=2)
        mamba_outs = [self.mamba_modules[i](chunk) for i, chunk in enumerate(x_chunks)]
        mamba_out = th.cat(mamba_outs, dim=2)
        # Attention mechanism
        attn_out, _ = self.attention(x, x, x)
        # Feature fusion: calculate gate values first, then fuse
        concat_for_gate = th.cat([tcn_out, mamba_out], dim=2)
        gate_values = self.gate(concat_for_gate)
        gated_features = gate_values * concat_for_gate
        combined = th.cat([gated_features, attn_out], dim=2)
        fused = self.fusion(combined)
        # Residual connection
        residual = self.residual_proj(x)
        # Output mapping
        out = self.mlp_out(fused + residual)
        
        # Smooth the generated signal (smoothing helps reduce noise fluctuations)
        if self.use_signal_smoothing:
            out = self.signal_smoother(out)
        
        return out, (None, None) if hid is None else hid

    @staticmethod
    def get_obj_value(criterion, out: TEN, lab: TEN, wup_dim: int) -> TEN:
        # Exclude warmup phase outputs when calculating loss
        return criterion(out, lab)[wup_dim:, :, :]


class RnnRegNet(MambaTCNRegNet):
    """
    Inherits from MambaTCNRegNet to maintain RNN interface compatibility
    """
    def __init__(self, inp_dim, mid_dim, out_dim, num_layers):
        # Call parent class initialization with default parameters
        super(RnnRegNet, self).__init__(
            inp_dim=inp_dim, 
            mid_dim=mid_dim, 
            out_dim=out_dim, 
            num_layers=num_layers,
            use_signal_smoothing=True,  # Enable signal smoothing by default
            smoothing_kernel=5,         # Default smoothing window size
            use_leaky_relu=True        # Use Tanh instead of LeakyReLU by default
        )


# ------------------------- Test Functions ------------------------- #

def check_mamba_tcn_reg_net():
    seq_len = 3600
    batch_size = 3
    inp_dim = 10
    mid_dim = 16
    out_dim = 8
    num_layers = 2

    net = MambaTCNRegNet(
        inp_dim=inp_dim, mid_dim=mid_dim, out_dim=out_dim, num_layers=num_layers,
        use_signal_smoothing=True, smoothing_kernel=5, use_leaky_relu=False
    )
    inp = th.randn(seq_len, batch_size, inp_dim)
    lab = th.randn(seq_len, batch_size, out_dim)

    out, _ = net(inp)
    print("输出 shape:", out.shape)
    
    # 计算并打印信号统计信息
    signal_mean = out.mean().item()
    signal_std = out.std().item()
    # 计算简单的 lag-1 自相关性
    diff = out[1:] - out[:-1]
    autocorr = (diff[:-1] * diff[1:]).mean().item()
    print(f"信号均值: {signal_mean:.6f}, 标准差: {signal_std:.6f}, lag-1自相关: {autocorr:.6f}")

    loss_func = nn.MSELoss()
    opt = th.optim.Adam(net.parameters(), lr=1e-3)

    for i in range(10):
        out, _ = net(inp)
        loss = loss_func(out, lab)
        print(f"Step {i}: Loss = {loss.item():.6f}")
        opt.zero_grad()
        loss.backward()
        opt.step()


def check_rnn_reg_net():
    seq_len = 3600
    batch_size = 3
    inp_dim = 10
    mid_dim = 16
    out_dim = 8
    num_layers = 2

    net = RnnRegNet(
        inp_dim=inp_dim, mid_dim=mid_dim, out_dim=out_dim, num_layers=num_layers
    )
    inp = th.randn(seq_len, batch_size, inp_dim)
    lab = th.randn(seq_len, batch_size, out_dim)

    out, _ = net(inp)
    print("输出 shape:", out.shape)
    loss_func = nn.MSELoss()
    opt = th.optim.Adam(net.parameters(), lr=1e-3)

    for i in range(10):
        out, _ = net(inp)
        loss = loss_func(out, lab)
        print(f"Step {i}: Loss = {loss.item():.6f}")
        opt.zero_grad()
        loss.backward()
        opt.step()


def check_rnn_in_real_trading():
    seq_length = 24
    input_size = 10

    rnn = MambaTCNRegNet(input_size, 20, 1, 1)
    input_seq = th.randn(seq_length, 1, input_size)
    hidden_state = None
    output_seq = []
    
    for t in range(seq_length):
        input_t = input_seq[t, :, :].unsqueeze(0)
        output_t, hidden_state = rnn(input_t, hidden_state)
        output_seq.append(output_t)
    
    output_seq = th.cat(output_seq, dim=0)
    output_seq2, _ = rnn(input_seq)
    
    print("输出序列 shape:", output_seq.shape)
    diff = th.abs(output_seq - output_seq2)
    print("差异:", diff)
    print("平均差异:", diff.mean())


if __name__ == "__main__":
    check_mamba_tcn_reg_net()
    # 可根据需要取消注释以下测试函数
    check_rnn_reg_net()
    check_rnn_in_real_trading()
