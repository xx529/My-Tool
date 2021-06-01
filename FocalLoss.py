# 二分类 FocalLoss

class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, y_pre, y_true):
        loss_l = y_true * torch.pow(1 - y_pre, self.gamma) * torch.log1p(y_pre)
        loss_r = (1 - y_true) * torch.pow(y_pre, self.gamma) * torch.log1p(1 - y_pre)
        loss = torch.mean(loss_l + loss_r)
        return loss