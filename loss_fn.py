import torch
import torch.nn as nn

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat


def calc_content_loss(input, target, mask=None):
    if mask != None:
        return nn.MSELoss()(mean_variance_norm(input*(1 - mask)), mean_variance_norm(target*(1 - mask)))
    else:
        return nn.MSELoss()(mean_variance_norm(input), mean_variance_norm(target))


def calc_style_loss(input, target):
    input_mean, input_std = calc_mean_std(input)
    target_mean, target_std = calc_mean_std(target)
    return nn.MSELoss()(input_mean, target_mean) + nn.MSELoss()(input_std, target_std)