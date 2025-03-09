"""注意力机制和位置编码模块"""

import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """位置编码
    
    为输入序列添加位置信息。
    
    参数:
        d_model (int): 输入特征维度
        max_len (int): 最大序列长度
        dropout (float): dropout比率
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 计算位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """前向传播
        
        参数:
            x (Tensor): 输入序列, shape (L, N, E)
            
        返回:
            output (Tensor): 添加位置编码后的序列, shape (L, N, E)
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class MultiheadAttention(nn.Module):
    """多头注意力层
    
    将输入投影到查询、键和值空间，执行缩放点积注意力，最后将多个头的输出拼接。
    
    参数:
        d_model (int): 输入特征维度
        nhead (int): 注意力头数
        dropout (float): dropout比率
    """
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0
        
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        
        # 线性投影层
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        """前向传播
        
        参数:
            query (Tensor): 查询序列, shape (L, N, E)
            key (Tensor): 键序列, shape (S, N, E)
            value (Tensor): 值序列, shape (S, N, E)
            attn_mask (Tensor): 注意力掩码, shape (L, S)
            key_padding_mask (Tensor): 键值对掩码, shape (N, S)
            
        返回:
            output (Tensor): 注意力输出, shape (L, N, E)
            attn_weights (Tensor): 注意力权重, shape (N, L, S)
        """
        batch_size = query.size(1)
        
        # 线性投影
        q = self.q_proj(query).view(-1, batch_size * self.nhead, self.d_k).transpose(0, 1)
        k = self.k_proj(key).view(-1, batch_size * self.nhead, self.d_k).transpose(0, 1)
        v = self.v_proj(value).view(-1, batch_size * self.nhead, self.d_k).transpose(0, 1)
        
        # 缩放点积注意力
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 应用掩码
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1), float('-inf'))
            
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 注意力输出
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(query.size(0), batch_size, self.d_model)
        attn_output = self.out_proj(attn_output)
        
        return attn_output, attn_weights.view(batch_size, self.nhead, -1, scores.size(-1)).mean(dim=1)