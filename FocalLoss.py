import torch.nn.functional as F
import torch.nn as nn


# 二分类 
class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
    
    # 输入 y_pre, y_ture 均为概率
    def forward(self, y_pre, y_true):  
        loss_l = y_true * torch.pow(1 - y_pre, self.gamma) * torch.log(y_pre)
        loss_r = (1 - y_true) * torch.pow(y_pre, self.gamma) * torch.log(1 - y_pre)
        loss = torch.sum(loss_l + loss_r)
        return loss

    
# 多分类
class MultiCEFocalLoss(nn.Module):
    def __init__(self, n_class, gamma, alpha=None):
        super(MultiCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = torch.ones(n_class) if alpha is None else alpha
        self.n_class = n_class

    # y_pre softmax后的概率，y_ture 是 index
    def forward(self, y_pre, y_true):
        mask = F.one_hot(y_true, self.n_class)
        idx = y_true.view(-1,1)
        alpha = self.alpha[idx]
        probs = (mask * y_pre).sum(1).view(-1, 1)
        log_p = probs.log()
        loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        return loss
