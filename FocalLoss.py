# 二分类 

class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, y_pre, y_true):
        loss_l = y_true * torch.pow(1 - y_pre, self.gamma) * torch.log(y_pre)
        loss_r = (1 - y_true) * torch.pow(y_pre, self.gamma) * torch.log(1 - y_pre)
        loss = torch.sum(loss_l + loss_r)
        return loss

# 多分类
