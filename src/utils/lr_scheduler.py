from torch.optim.lr_scheduler import _LRScheduler

class PolyLR(_LRScheduler):
    """
    This class implements the polynomial learning rate schedule.
    """
    def __init__(self, optimizer, T_max, eta_min=0, power=0.9, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        self.power = power
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) *
                (1 - self.last_epoch / self.T_max) ** self.power
                for base_lr in self.base_lrs]
