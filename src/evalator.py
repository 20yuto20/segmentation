import numpy as np


class Evaluator(object):
    def __init__(self, num_class, ignore_iabel):
        
        self.num_class = num_class
        self.ignore_label = ignore_iabel
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image != self.ignore_label)
        gt_image = gt_image[mask]
        pre_image = pre_image[mask]
        
        label = self.num_class * gt_image.astype('int') + pre_image
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def debug_info(self):
        print("Confusion Matrix:")
        print(self.confusion_matrix)
        print(f"Total pixels: {np.sum(self.confusion_matrix)}")
        print(f"Correct pixels: {np.sum(np.diag(self.confusion_matrix))}")