import torch
import numpy as np

class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.reset()

    def reset(self):
        self.total_area_inter = torch.zeros(self.num_class).cuda()
        self.total_area_union = torch.zeros(self.num_class).cuda()
        self.total_area_pred = torch.zeros(self.num_class).cuda()
        self.total_area_label = torch.zeros(self.num_class).cuda()

    def add_batch(self, predict, label):
        area_inter, area_union, area_pred, area_label = self.intersectionAndUnionGPU(predict, label, self.num_class)
        self.total_area_inter += area_inter
        self.total_area_union += area_union
        self.total_area_pred += area_pred
        self.total_area_label += area_label

    def intersectionAndUnionGPU(self, output, target, K, ignore_index=255):
        output = output.clone()
        target = target.clone()
        output[target == ignore_index] = ignore_index
        intersection = output[output == target]
        area_inter = torch.histc(intersection, bins=K, min=0, max=K-1)
        area_pred = torch.histc(output, bins=K, min=0, max=K-1)
        area_label = torch.histc(target, bins=K, min=0, max=K-1)
        area_union = area_pred + area_label - area_inter
        return area_inter, area_union, area_pred, area_label

    def Mean_Intersection_over_Union(self):
        MIoU = self.total_area_inter / (self.total_area_union + 1e-10)
        MIoU = MIoU.cpu().numpy()
        return np.mean(MIoU)

    # def Pixel_Accuracy(self):
    #     Acc = torch.sum(self.total_area_inter) / (torch.sum(self.total_area_label) + 1e-10)
    #     Acc = Acc.cpu().numpy()
    #     return Acc
    
    def Pixel_Accuracy(self):
        Acc = torch.sum(self.total_area_inter) / (torch.sum(self.total_area_pred) + 1e-10)
        Acc = Acc.cpu().numpy()
        return Acc