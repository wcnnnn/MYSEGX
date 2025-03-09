"""优化器模块"""

import math
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

class SGD(Optimizer):
    """随机梯度下降优化器
    
    参数:
        params: 待优化参数
        lr: 学习率
        momentum: 动量因子
        weight_decay: 权重衰减
        nesterov: 是否使用Nesterov动量
    """
    def __init__(self, params, lr=0.01, momentum=0, weight_decay=0, nesterov=False):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
            
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            nesterov = group['nesterov']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                d_p = p.grad.data
                
                if weight_decay != 0:
                    d_p = d_p.add(p.data, alpha=weight_decay)
                    
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p)
                        
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf
                        
                p.data.add_(d_p, alpha=-group['lr'])
                
        return loss

class Adam(Optimizer):
    """Adam优化器
    
    参数:
        params: 待优化参数
        lr: 学习率
        betas: 一阶和二阶矩估计的指数衰减率
        eps: 数值稳定性常数
        weight_decay: 权重衰减
        amsgrad: 是否使用AMSGrad变体
    """
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
            
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad.data
                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if group['amsgrad']:
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)
                        
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])
                    
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                if group['amsgrad']:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
                
        return loss

class CosineAnnealingLR(_LRScheduler):
    """余弦退火学习率调度器
    
    参数:
        optimizer: 优化器
        T_max: 最大迭代次数
        eta_min: 最小学习率
        last_epoch: 上一轮的epoch
    """
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                for base_lr in self.base_lrs]

class StepLR(_LRScheduler):
    """按步长调整学习率的调度器
    
    参数:
        optimizer: 优化器
        step_size: 学习率调整步长
        gamma: 学习率调整系数
        last_epoch: 上一轮的epoch
    """
    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self):
        return [base_lr * self.gamma ** (self.last_epoch // self.step_size)
                for base_lr in self.base_lrs]