"""Contains the loss functions."""

import torch
from torch import nn
from loss_fn import calc_content_loss, calc_style_loss


class MultiTaskLoss(nn.Module):
    """Computes and combines the losses for the three tasks.
    Has two modes:
    1) 'fixed': the losses multiplied by fixed weights and summed
    2) 'learned': we learn the losses, not implemented...
    """

    def __init__(self, loss_type="learned", loss_uncertainties=(1.0, 1.0), enabled_tasks=(True, True)):
        """Creates a new instance.
        :param loss_type Either 'fixed' or 'learned'
        :param loss_uncertainties A 3 tuple of (semantic seg uncertainty, instance seg uncertainty,
        depth uncertainty). If 'fixed' then these should be floats, if 'learned' then they should be
        torch Parameters.
        """
        super().__init__()

        self.loss_type = loss_type
        self.loss_uncertainties = loss_uncertainties
        self.enabled_tasks = enabled_tasks

        self.content_loss = calc_content_loss
        self.style_loss = calc_style_loss
        self.mseloss = nn.MSELoss()
        self.bceloss = nn.BCELoss()


    def domain_loss(self, x1, x2, mask, x1_gram, x2_gram,  x1_domain, x2_domain):
        return self.content_loss(x1, x1_domain) + self.content_loss(x2, x2_domain) + self.content_loss(x1_domain, x2_domain, mask.unsqueeze(1)) + \
               self.style_loss(x1_domain, x2_domain) + \
               self.mseloss(x1_gram, x2_gram)

    def change_loss(self, predict, mask):
        return self.bceloss(predict.squeeze(1), mask)


    def calculate_total_loss(self, *losses):
        domain_loss, change_loss = losses
        domain_uncertainty, change_uncertainty = self.loss_uncertainties
        domain_enabled, change_enabled = self.enabled_tasks

        loss = 0

        if self.loss_type == 'learned':
            if domain_enabled:
                loss += 0.5 / (domain_uncertainty ** 2) * domain_loss + torch.log(1 + domain_uncertainty ** 2)
            if change_enabled:
                loss += 0.5 / (change_uncertainty ** 2) * change_loss + torch.log(1 + change_uncertainty ** 2)

        else:
            raise ValueError

        return loss

    def forward(self, x1, x2, mask, x1_domain, x2_domain, x1_gram, x2_gram, predict):
        doamin_enabled, change_enabled = self.enabled_tasks

        domain_loss = self.domain_loss(x1, x2, mask, x1_gram, x2_gram, x1_domain, x2_domain) if doamin_enabled else None
        change_loss = self.change_loss(predict, mask) if change_enabled else None

        total_loss = self.calculate_total_loss(domain_loss, change_loss)

        domain_loss_item = domain_loss.item() if domain_loss is not None else 0
        change_loss_item = change_loss.item() if change_loss is not None else 0

        return total_loss, domain_loss, change_loss