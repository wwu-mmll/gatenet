from fastai.basics import nn, torch, Module, Tensor


class GateNet(Module):
    """Neural network architecture for automated flow cytometry gating"""
    def __init__(self, hparams: dict):
        super(GateNet, self).__init__()
        nm = hparams['n_markers']
        ng = hparams['n_gates']
        nf = hparams['n_filter']
        nfc = hparams['n_filter_context']
        nh = hparams['n_head_nodes']

        self.conv_module = conv_block(nm, nf)
        self.conv_module_context = conv_block(nm, nfc)
        self.head = head_block(nf[-1] + nfc[-1], nh, ng)

    def forward(self, x: Tensor, x_c: Tensor, single_sample=False) -> Tensor:
        if x.shape[0] == 1:
            single_sample = True
            x = x.repeat(2, 1)
            x_c = x_c.repeat(2, 1, 1)
        x_s = x.reshape(x.shape[0], 1, 1, x.shape[1])
        x_s = self.conv_module(x_s)
        x_s = nn.Flatten()(x_s)
        x_c = x_c.reshape(x_c.shape[0], 1, x_c.shape[1], x_c.shape[2])
        x_c = self.conv_module_context(x_c)  # self.conv_module(x_c)
        x_c = nn.AvgPool2d(kernel_size=(x_c.shape[-2], 1))(x_c)
        x_c = nn.Flatten()(x_c)
        x_merged = torch.cat((x_s, x_c), dim=1)
        x = self.head(x_merged)
        if single_sample:
            x = x[:1]
        return x


def conv_block(input_dim: int, nf: tuple, nf_in=1, act=nn.ReLU) -> nn.Sequential:
    conv_mod = []
    for i in range(len(nf)):
        nf_in = nf[i - 1] if i > 0 else nf_in
        ks_1 = input_dim if i == 0 else 1
        conv_layer = [nn.Conv2d(nf_in, nf[i], kernel_size=(1, ks_1)), nn.BatchNorm2d(nf[i]), act()]
        conv_mod += conv_layer
    return nn.Sequential(*conv_mod)


def head_block(input_dim: int, nh: tuple, ng: int, act=nn.ReLU) -> nn.Sequential:
    n_nodes = [input_dim] + list(nh) + [ng]
    head_mod = []
    for i in range(len(n_nodes) - 1):
        head_layer = [nn.Linear(n_nodes[i], n_nodes[i + 1]), nn.BatchNorm1d(n_nodes[i + 1]), act()]
        if i == len(nh):
            head_layer = head_layer[:-1] + [nn.Softmax(dim=1)]
        head_mod += head_layer
    return nn.Sequential(*head_mod)
