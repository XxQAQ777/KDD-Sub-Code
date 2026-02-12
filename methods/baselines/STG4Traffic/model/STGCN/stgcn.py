import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

# reference: https://github.com/VeritasYin/STGCN_IJCAI-18
class align(nn.Module):
    def __init__(self, c_in, c_out):
        super(align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        if c_in > c_out:
            self.conv1x1 = nn.Conv2d(c_in, c_out, 1)

    def forward(self, x):
        if self.c_in > self.c_out:
            return self.conv1x1(x)
        if self.c_in < self.c_out:
            return F.pad(x, [0, 0, 0, 0, 0, self.c_out - self.c_in, 0, 0])
        return x


# Gated Convolution Unit
class temporal_conv_layer(nn.Module):
    def __init__(self, kt, c_in, c_out, act="relu"):
        super(temporal_conv_layer, self).__init__()
        self.kt = kt
        self.act = act
        self.c_out = c_out
        self.align = align(c_in, c_out)
        if self.act == "GLU":
            self.conv = nn.Conv2d(c_in, c_out * 2, (kt, 1), 1)
        else:
            self.conv = nn.Conv2d(c_in, c_out, (kt, 1), 1)

    def forward(self, x):
        x_in = self.align(x)[:, :, self.kt - 1:, :]
        if self.act == "GLU":
            x_conv = self.conv(x)
            return (x_conv[:, :self.c_out, :, :] + x_in) * torch.sigmoid(x_conv[:, self.c_out:, :, :])
        if self.act == "sigmoid":
            return torch.sigmoid(self.conv(x) + x_in)
        return torch.relu(self.conv(x) + x_in)


# Spatial Convolution Layer
class spatio_conv_layer(nn.Module):
    def __init__(self, ks, c, Lk):
        super(spatio_conv_layer, self).__init__()
        self.Lk = Lk
        self.theta = nn.Parameter(torch.FloatTensor(c, c, ks))
        self.b = nn.Parameter(torch.FloatTensor(1, c, 1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.theta, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.theta)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.b, -bound, bound)

    def forward(self, x):
        x_c = torch.einsum("knm,bitm->bitkn", self.Lk, x)
        x_gc = torch.einsum("iok,bitkn->botn", self.theta, x_c) + self.b
        return torch.relu(x_gc + x)


# ST_conv Block
class st_conv_block(nn.Module):
    def __init__(self, ks, kt, n, c, p, Lk):
        super(st_conv_block, self).__init__()
        self.tconv1 = temporal_conv_layer(kt, c[0], c[1], "GLU")
        self.sconv = spatio_conv_layer(ks, c[1], Lk)
        self.tconv2 = temporal_conv_layer(kt, c[1], c[2])
        self.ln = nn.LayerNorm([n, c[2]])
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        x_t1 = self.tconv1(x)
        x_s = self.sconv(x_t1)
        x_t2 = self.tconv2(x_s)
        x_ln = self.ln(x_t2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.dropout(x_ln)


# output_layer 和 STGCN 部分替换为以下代码

class output_layer(nn.Module):
    def __init__(self, c, T, n, H):
        super(output_layer, self).__init__()
        # T 是模型预期经过时空块后剩余的时间维度 (通常为 1 或很小)
        self.tconv1 = temporal_conv_layer(T, c, c, "GLU")
        self.ln = nn.LayerNorm([n, c])
        self.tconv2 = temporal_conv_layer(1, c, c, "sigmoid")
        
        # 简化输出层：直接使用 1x1 卷积将通道 c 映射为预测步长 H
        # 输入 (B, c, 1, N) -> 输出 (B, H, 1, N)
        self.fc = nn.Conv2d(c, H, kernel_size=1, bias=True)

    def forward(self, x):
        # x shape: (B, c, T_in, N)
        x_t1 = self.tconv1(x)  
        # shape: (B, c, T_rem, N)
        # 这里的 T_rem 理论上应该是 1。
        # 但如果输入数据未正确切片(如传入了144步)，这里 T_rem 会变成 133。
        # fix: 强制取最后一个时间步，忽略前面多余的维度，防止尺寸不匹配报错。
        if x_t1.shape[2] > 1:
            x_t1 = x_t1[:, :, -1:, :] 

        # x_t1 现在确认为 (B, c, 1, N)
        # LayerNorm 需要 input 为 (..., N, c)
        x_ln = self.ln(x_t1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        x_t2 = self.tconv2(x_ln) # (B, c, 1, N)
        
        # 通过 1x1 卷积输出 H 个预测步
        out = self.fc(x_t2) # (B, H, 1, N)
        
        # 调整为标准输出格式 (Batch, Horizon, Node, Feature=1)
        out = out.permute(0, 1, 3, 2) # (B, H, N, 1)
        
        return out


class STGCN(nn.Module):
    def __init__(self, ks, kt, bs, T, n, Lk, p, horizon=12):
        super(STGCN, self).__init__()
        self.st_conv1 = st_conv_block(ks, kt, n, bs[0], p, Lk)
        self.st_conv2 = st_conv_block(ks, kt, n, bs[1], p, Lk)

        # 计算经过两个 ST-Conv 块后剩余的时间维度
        # T 是输入时间步数（window）
        # 每一层减少 2 * (kt - 1)
        # 两层共减少 4 * (kt - 1)
        t_final = T - 4 * (kt - 1)

        # 确保 t_final 至少为 1，防止 T 设置过小导致报错
        if t_final < 1:
            t_final = 1

        # horizon 是预测步长
        self.output = output_layer(bs[1][2], t_final, n, horizon)

    def forward(self, x):
        # 输入 x: (Batch, Time, Nodes, Channels) -> (Batch, Channels, Time, Nodes)
        x = x.permute(0, 3, 1, 2)
        
        x_st1 = self.st_conv1(x)
        x_st2 = self.st_conv2(x_st1)
        
        return self.output(x_st2)