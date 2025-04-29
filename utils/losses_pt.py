import torch


def soft_dice_loss(pred, gt, epsilon=1e-5):
    intersection = torch.sum(pred * gt)
    numerator = 2.0 * intersection

    pred_squared = torch.sum(torch.square(pred))
    gt_squared = torch.sum(torch.square(gt))
    denominator = pred_squared + gt_squared

    dice = numerator / (denominator + epsilon)

    loss = 1.0 - dice
    return loss
