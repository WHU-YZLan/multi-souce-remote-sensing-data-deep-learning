import numpy as np
import cv2

def compute_iou(pr, gt, class_num):
    ious = np.zeros(class_num)
    for cls in range(1, class_num):
        pred_inds = pr == cls
        target_inds = gt == cls
        intersection = (pred_inds[target_inds]).sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        if union > 0:
            ious[cls] = intersection / union
    return ious

#计算单张图像miou
def compute_miou(pr, gt, class_num):
    ious = compute_iou(pr, gt, class_num)
    miou = np.mean(ious[1:5])  # 忽略背景类的IoU
    return miou

#计算若干张图像miou
def compute_m_miou(pr_list, gt_list, class_num):
    miou = 0.
    for i in range(len(pr_list)):
        ious = compute_iou(pr_list[i], gt_list[i], class_num)
        miou += np.mean(ious[1:])
    miou /= len(pr_list)
    return miou

# 示例
if __name__ == '__main__':
    pr = cv2.imread("predict/pre.tif", 0)
    gt = cv2.imread("gt/up_label.tif", 0)
    class_num = 6
    ious = compute_iou(pr, gt, class_num)
    miou = compute_miou(pr, gt, class_num)
    print("每个类别的IoU:", ious)
    print("mIoU:", miou)
