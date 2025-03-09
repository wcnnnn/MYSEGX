"""Transformer编码器和解码器块模块"""

import torch
import torch.nn as nn

class TransformerEncoderBlock(nn.Module):
    """Transformer编码器块
    
    包含多头自注意力层和前馈网络层。
    
    参数:
        d_model (int): 输入特征维度
        nhead (int): 注意力头数
        dim_feedforward (int): 前馈网络隐藏层维度
        dropout (float): dropout比率
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        
        # 多头自注意力层
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # 前馈网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """前向传播
        
        参数:
            src (Tensor): 输入序列, shape (L, N, E)
            src_mask (Tensor): 注意力掩码, shape (L, L)
            src_key_padding_mask (Tensor): 键值对掩码, shape (N, L)
            
        返回:
            output (Tensor): 输出序列, shape (L, N, E)
        """
        # 多头自注意力
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # 前馈网络
        src2 = self.linear2(self.dropout(torch.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src

class TransformerDecoderBlock(nn.Module):
    """Transformer解码器块
    
    包含多头自注意力层、多头交叉注意力层和前馈网络层。
    
    参数:
        d_model (int): 输入特征维度
        nhead (int): 注意力头数
        dim_feedforward (int): 前馈网络隐藏层维度
        dropout (float): dropout比率
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        
        # 多头自注意力层
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # 多头交叉注意力层
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # 前馈网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """前向传播
        
        参数:
            tgt (Tensor): 目标序列, shape (T, N, E)
            memory (Tensor): 记忆序列, shape (S, N, E)
            tgt_mask (Tensor): 目标注意力掩码, shape (T, T)
            memory_mask (Tensor): 记忆注意力掩码, shape (T, S)
            tgt_key_padding_mask (Tensor): 目标键值对掩码, shape (N, T)
            memory_key_padding_mask (Tensor): 记忆键值对掩码, shape (N, S)
            
        返回:
            output (Tensor): 输出序列, shape (T, N, E)
        """
        # 多头自注意力
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # 多头交叉注意力
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # 前馈网络
        tgt2 = self.linear2(self.dropout(torch.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt