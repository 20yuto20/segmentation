import torch
import numpy as np

class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.reset()

    def reset(self):
        self.total_area_inter = np.zeros(self.num_class)
        self.total_area_union = np.zeros(self.num_class)
        self.total_area_pred = np.zeros(self.num_class)
        self.total_area_label = np.zeros(self.num_class)

    # def add_batch(self, predict, label):
    #     assert predict.shape == label.shape
    #     area_inter, area_union, area_pred, area_label = self.intersectionAndUnion(predict, label, self.num_class)
    #     self.total_area_inter += area_inter
    #     self.total_area_union += area_union
    #     self.total_area_pred += area_pred
    #     self.total_area_label += area_label
    
    def add_batch(self, predict, label):
        try:
            assert predict.shape == label.shape, f"Shape mismatch: predict {predict.shape}, label {label.shape}"
            area_inter, area_union, area_pred, area_label = self.intersectionAndUnion(predict, label, self.num_class)
            self.total_area_inter += area_inter
            self.total_area_union += area_union
            self.total_area_pred += area_pred
            self.total_area_label += area_label
        except AssertionError as e:
            print(f"AssertionError in add_batch: {e}")
            raise

    def intersectionAndUnion(self, output, target, K, ignore_index=255):
        # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
        assert (output.ndim in [1, 2, 3])
        assert output.shape == target.shape
        output = output.reshape(-1)
        target = target.reshape(-1)
        output[target == ignore_index] = ignore_index
        intersection = output[output == target]
        area_inter = np.histogram(intersection, bins=K, range=(0, K-1))[0]
        area_output = np.histogram(output, bins=K, range=(0, K-1))[0]
        area_target = np.histogram(target, bins=K, range=(0, K-1))[0]
        area_union = area_output + area_target - area_inter
        return area_inter, area_union, area_output, area_target

    def Mean_Intersection_over_Union(self):
        MIoU = self.total_area_inter / (self.total_area_union + 1e-10)
        return np.mean(MIoU)

    def Pixel_Accuracy(self):
        Acc = np.sum(self.total_area_inter) / (np.sum(self.total_area_pred) + 1e-10)
        return Acc