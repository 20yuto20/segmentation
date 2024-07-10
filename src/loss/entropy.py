import torch.nn as nn


# TODO: 損失関数としてクロスエントロピーの実装（cfgの受け取りと、sugget.pyでの呼び出し、ignore_indexの機能確認）
class CrossEntropyLoss(nn.Module, cfg):
    def __init__(self, ignore_index=cfg.dataset.ignore_label):
        super(CrossEntropyLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, pred, target):
        return self.ce_loss(pred, target)