from torchvision.models.vgg import vgg19
import torch.nn as nn
import torch


def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)
                                           ).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    return loss, nTotal.item()
