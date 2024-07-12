import torch.nn as nn



class CrossEntropyLoss(nn.Module):
    def __init__(self, cfg):
        super(CrossEntropyLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=cfg.dataset.ignore_label)

    def forward(self, pred, target):
        return self.ce_loss(pred, target)
    

# 損失関数の設定
class PSPLoss(nn.Module):
    """PSPNetの損失関数のクラスです。"""

    def __init__(self, cfg):
        super(PSPLoss, self).__init__()
        self.aux_weight = cfg.optimizer.loss.aux_weight  # aux_lossの重み
        self.criterion = CrossEntropyLoss(cfg)

    def forward(self, outputs, targets):
        """
        損失関数の計算。

        Parameters
        ----------
        outputs : PSPNetの出力(tuple)
            (output=torch.Size([num_batch, 21, 475, 475]), output_aux=torch.Size([num_batch, 21, 475, 475]))。

        targets : [num_batch, 475, 475]
            正解のアノテーション情報

        Returns
        -------
        loss : テンソル
            損失の値
        """

        loss = self.criterion(outputs[0], targets)
        loss_aux = self.criterion(outputs[1], targets)

        return loss+self.aux_weight*loss_aux

