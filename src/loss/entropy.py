import torch.nn as nn



class CrossEntropyLoss(nn.Module):
    def __init__(self, cfg):
        super(CrossEntropyLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=cfg.dataset.ignore_label)

    def forward(self, pred, target):
        return self.ce_loss(pred, target)