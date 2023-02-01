from torch import nn, Tensor
# import torch.nn.functional as F
# import torch


def jaccard_coef(y_pred: Tensor, y_true: Tensor, smooth: float = 1) -> Tensor:
  
    y_pred = nn.Sigmoid()(y_pred)

    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)

    intersection = (y_pred * y_true).sum()
    union = y_true.sum() + y_pred.sum()
    jaccard = (intersection + smooth) / (union + smooth - intersection)
    return jaccard


def dice_coef(y_pred: Tensor, y_true: Tensor, smooth: float = 1) -> Tensor:

    y_pred = nn.Sigmoid()(y_pred)

    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    
    intersection = (y_pred * y_true).sum()
    union = y_pred.sum() + y_true.sum()
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def __call__(self, y_pred, y_true, smooth=1):
        loss = 1. - dice_coef(y_pred, y_true, smooth)
        return loss
  
# def diceCoeff(pred, gt, eps = 1e-5):
 
#     # activation_func = nn.Tanh()
#     activation_func = nn.Sigmoid()

#     pred = activation_func(pred)

#     n = gt.size(0)
#     pred_flat = pred.view(n, -1)
#     gt_flat = gt.view(n, -1)

#     tp = torch.sum(gt_flat * pred_flat, dim = 1)
#     fp = torch.sum(pred_flat, dim = 1) - tp
#     fn = torch.sum(gt_flat, dim = 1) - tp

#     loss = (2 * tp + eps) / (2 * tp + fp + fn + eps)

#     return loss.sum() / n

# class DiceLoss(nn.Module):
#     def __init__(self):
#         super(DiceLoss, self).__init__()

#     def	forward(self, input, target):
#         return 1- diceCoeff(input, target)


# class FocalLoss(nn.Module):
#   def __init__(self, alpha=1, gamma=2, logits=True, reduce=True):
#     super(FocalLoss, self).__init__()
#     self.alpha = alpha
#     self.gamma = gamma
#     self.logits = logits
#     self.reduce = reduce

#   def forward(self, inputs, targets):
#     if self.logits:
#       BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
#     else:
#       BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
#     pt = torch.exp(-BCE_loss)
#     F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

#     if self.reduce:
#       return torch.mean(F_loss)
#     else:
#       return F_loss


# class SigmoidFocalLoss(nn.Module):

#   def __init__(self, gamma=2.0, alpha=0.25, reduction='mean'):
#     super(SigmoidFocalLoss).__init__()
#     self.gamma = gamma
#     self.alpha = alpha
#     self.reduction = reduction  

#   def __call__(self, input: torch.Tensor, target: torch.Tensor) -> Tensor:
#     target = target.type(input.type())

#     logpt = -F.binary_cross_entropy_with_logits(input, target, reduction='none')
#     pt = torch.exp(logpt)

#     # compute the loss
#     loss = -((1 - pt).pow(self.gamma)) * logpt

#     if self.alpha is not None:
#       loss = loss * (self.alpha * target + (1 - self.alpha) * (1 - target))

#     if self.reduction == 'mean':
#       loss = loss.mean()
#     if self.reduction == 'sum':
#       loss = loss.sum()
#     if self.reduction == 'batchwise_mean':
#       loss = loss.sum(0)

#     return loss