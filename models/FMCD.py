import torch
from torch import nn
import torchvision
from models.layers import MixAttentionBlock, PixelwiseLinear, MixingBlock, Up, DomainUp, gram_matrix
from torch.nn import Module, ModuleList, Sigmoid
from models.chornet import hornet


class ChangeClassifier(Module):
    def __init__(
        self,
            bkbn_name="efficientnet_b4", pretrained=True, output_layer_bkbn="4", freeze_backbone=False,
            gnc_dim=112, gnc_order=3,
    ):
        super().__init__()

        # Load the pretrained backbone according to parameters:
        self._backbone1 = _get_backbone(
            bkbn_name, pretrained, output_layer_bkbn, freeze_backbone
        )
        self._backbone2 = _get_backbone(
            bkbn_name, pretrained, output_layer_bkbn, freeze_backbone
        )

        self.gnconv1 = hornet(gnc_dim, gnc_order)
        self.gnconv2 = hornet(gnc_dim, gnc_order)

        self.mixing = MixingBlock(gnc_dim * 2, gnc_dim)

        self.MixBlcok = ModuleList(
            [
                MixingBlock(112, 56),
                MixingBlock(64, 32),
                MixingBlock(48, 24),
            ]
        )
        self.ChangeMix = ModuleList(
            [
                MixAttentionBlock(dim=56),
                MixAttentionBlock(dim=32),
                MixAttentionBlock(dim=24),
            ]
        )
        self.Upsample = ModuleList(
            [
                Up(112, 56),
                Up(56, 32),
                Up(32, 24),
            ]
        )
        self.finalupsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.classify = PixelwiseLinear([24, 12, 6], [12, 6, 1], Sigmoid())

        # domain decoder
        self.domain_upsample = ModuleList(
            [
                DomainUp(112, 56),
                DomainUp(56, 32),
                DomainUp(32, 16),
                DomainUp(16, 3)
            ]
        )

        self.gram = gram_matrix

        # loss params
        self.domain_log_var = nn.Parameter(torch.tensor(1.0, dtype=torch.float))
        self.change_log_var = nn.Parameter(torch.tensor(1.0, dtype=torch.float))

    def cnn(self, x1, x2):
        x1_downsample = []
        x2_downsample = []

        for num, layer in enumerate(self._backbone1):
            x1 = layer(x1)
            if num != 0:
                x1_downsample.append(x1)
        for num, layer in enumerate(self._backbone2):
            x2 = layer(x2)
            if num != 0:
                x2_downsample.append(x2)

        return x1_downsample, x2_downsample

    def decoder(self, x1, x2, x1_downsample, x2_downsample):
        x = self.mixing(x1, x2)
        for i in range(3):
            x = self.Upsample[i](x)
            mix, attn = self.ChangeMix[i](x1_downsample[2-i], x2_downsample[2-i])
            x = self.MixBlcok[i](x, mix) * attn
        x = self.finalupsample(x)

        return x

    def domain_decoder(self, x):
        for i in range(4):
            x = self.domain_upsample[i](x)
        return x

    def get_loss_params(self) -> (nn.Parameter, nn.Parameter):
        """Returns sem_log_var, inst_log_var, depth_log_var"""
        return self.domain_log_var,  self.change_log_var

    def forward(self, x1, x2):
        x1_downsample, x2_downsample = self.cnn(x1, x2)

        x1 = self.gnconv1(x1_downsample[-1])
        x2 = self.gnconv2(x2_downsample[-1])

        # gram matric
        x1_gram = self.gram(x1_downsample[-1])
        x2_gram = self.gram(x2_downsample[-1])

        # domain decoder
        x1_domain = self.domain_decoder(x1)
        x2_domain = self.domain_decoder(x2)

        # change detection
        x = self.decoder(x1, x2, x1_downsample, x2_downsample)

        pred = self.classify(x)
        return pred, x1_domain, x2_domain, x1_gram, x2_gram


def _get_backbone(bkbn_name, pretrained, output_layer_bkbn, freeze_backbone) -> ModuleList:
    # The whole model:
    entire_model = getattr(torchvision.models, bkbn_name)(pretrained=pretrained).features

    # Slicing it:
    derived_model = ModuleList([])
    for name, layer in entire_model.named_children():
        derived_model.append(layer)
        if name == output_layer_bkbn:
            break

    # Freezing the backbone weights:
    if freeze_backbone:
        for param in derived_model.parameters():
            param.requires_grad = False
    return derived_model




if __name__ == '__main__':
    x1 = torch.randn(4, 3, 256, 256).cuda()
    x2 = x1
    net = ChangeClassifier().cuda()
    out,_,_,_,_ = net(x1, x2)
    print(out.shape)