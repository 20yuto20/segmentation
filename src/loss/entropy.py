import torch.nn as nn



class PSPLoss(nn.Module):
    def __init__(self, cfg):
        super(PSPLoss, self).__init__()
        self.aux_weight = cfg.optimizer.loss.aux_weight
        self.criterion = CrossEntropyLoss(cfg)

    def forward(self, outputs, targets):
        if isinstance(outputs, tuple):
            main_loss = self.criterion(outputs[0], targets)
            aux_loss = self.criterion(outputs[1], targets)
            return main_loss + self.aux_weight * aux_loss
        else:
            return self.criterion(outputs, targets)

class CrossEntropyLoss(nn.Module):
    def __init__(self, cfg):
        super(CrossEntropyLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=cfg.dataset.ignore_label)

    def forward(self, pred, target):
        # pred の形状が [B, C, H, W] であることを確認
        if pred.dim() == 4:
            # [B, C, H, W] -> [B, H, W, C]
            pred = pred.permute(0, 2, 3, 1)
        # [B, H, W, C] -> [B*H*W, C]
        pred = pred.contiguous().view(-1, pred.size(-1))
        # [B, H, W] -> [B*H*W]
        target = target.view(-1)
        return self.ce_loss(pred, target)