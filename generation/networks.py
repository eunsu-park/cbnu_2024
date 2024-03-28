import torch
import torch.nn as nn
import torch.nn.init as init


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels, out_channels, nb_layers, nb_feat_init, nb_feat_max, use_sigmoid):
        super(PatchDiscriminator, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nb_layers = nb_layers
        self.nb_feat_init = nb_feat_init
        self.nb_feat_max = nb_feat_max
        self.use_sigmoid = use_sigmoid
        self.build()

    def build(self):
        nb_feat_in = self.in_channels + self.out_channels
        nb_feat_out = self.nb_feat_init
        block = [nn.Conv2d(nb_feat_in, nb_feat_out, kernel_size=4, stride=2, padding=1, bias=True),
                 nn.LeakyReLU(0.2)]

        for _ in range(1, self.nb_layers):
            nb_feat_in = nb_feat_out
            nb_feat_out = min(nb_feat_out*2, self.nb_feat_max)
            block += [nn.Conv2d(nb_feat_in, nb_feat_out, kernel_size=4, stride=2, padding=1, bias=False),
                      nn.BatchNorm2d(nb_feat_out), nn.LeakyReLU(0.2)]

        nb_feat_in = nb_feat_out
        nb_feat_out = min(nb_feat_out*2, self.nb_feat_max)
        block += [nn.Conv2d(nb_feat_in, nb_feat_out, kernel_size=4, stride=1, padding=1, bias=False),
                  nn.BatchNorm2d(nb_feat_out), nn.LeakyReLU(0.2)]

        nb_feat_in = nb_feat_out
        nb_feat_out = 1
        block += [nn.Conv2d(nb_feat_in, nb_feat_out, kernel_size=4, stride=1, padding=1, bias=True)]
        if self.use_sigmoid :
            block += [nn.Sigmoid()]

        self.model = nn.Sequential(*block)
        
    def forward(self, x):
        return self.model(x)


class PixelDiscriminator(nn.Module):
    def __init__(self, in_channels, out_channels, nb_feat_init, use_sigmoid):
        super(PixelDiscriminator, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nb_feat_init = nb_feat_init
        self.use_sigmoid = use_sigmoid
        self.build()

    def build(self):
        nb_feat_in = self.in_channels + self.out_channels
        nb_feat_out = self.nb_feat_init
        block = [nn.Conv2d(nb_feat_in, nb_feat_out, kernel_size=1, stride=1, padding=0, bias=True),
                 nn.LeakyReLU(0.2)]
        
        nb_feat_in = nb_feat_out
        nb_feat_out = nb_feat_in * 2
        block += [nn.Conv2d(nb_feat_in, nb_feat_out, kernel_size=1, stride=1, padding=0, bias=False),
                  nn.BatchNorm2d(nb_feat_out), nn.LeakyReLU(0.2)]
        
        nb_feat_in = nb_feat_out
        nb_feat_out = 1
        block += [nn.Conv2d(nb_feat_in, nb_feat_out, kernel_size=1, stride=1, padding=0, bias=True)]
        if self.use_sigmoid :
            block += [nn.Sigmoid()]
        
        self.model = nn.Sequential(*block)
        
    def forward(self, x):
        return self.model(x)


class UnetDown(nn.Module):
    def __init__(self, nb_feat_in, nb_feat_out):
        super(UnetDown, self).__init__()
        self.build(nb_feat_in, nb_feat_out)

    def build(self, nb_feat_in, nb_feat_out):
        block = [nn.LeakyReLU(0.2),
                 nn.Conv2d(nb_feat_in, nb_feat_out, kernel_size=4, stride=2, padding=1, bias=False),
                 nn.BatchNorm2d(nb_feat_out)]
        self.model = nn.Sequential(*block)

    def forward(self, inp):
        return self.model(inp)
    

class UnetCenter(nn.Module):
    def __init__(self, nb_feat_in, nb_feat_out):
        super(UnetCenter, self).__init__()
        self.build(nb_feat_in, nb_feat_out)

    def build(self, nb_feat_in, nb_feat_out):
        block = [nn.LeakyReLU(0.2),
                 nn.Conv2d(nb_feat_in, nb_feat_out, kernel_size=4, stride=2, padding=1, bias=True),
                 nn.ReLU(),
                 nn.ConvTranspose2d(nb_feat_out, nb_feat_in, kernel_size=4, stride=2, padding=1, bias=False),
                 nn.BatchNorm2d(nb_feat_in),
                 nn.Dropout2d(0.5)]
        self.model = nn.Sequential(*block)

    def forward(self, inp):
        return self.model(inp)

class UnetUp(nn.Module):
    def __init__(self, nb_feat_in, nb_feat_out, use_dropout):
        super(UnetUp, self).__init__()
        self.build(nb_feat_in, nb_feat_out, use_dropout)

    def build(self, nb_feat_in, nb_feat_out, use_dropout):
        block = [nn.ReLU(),
                 nn.ConvTranspose2d(nb_feat_in, nb_feat_out, kernel_size=4, stride=2, padding=1, bias=False),
                 nn.BatchNorm2d(nb_feat_out)]
        if use_dropout == True :
                block += [nn.Dropout2d(0.5)]
        self.model = nn.Sequential(*block)

    def forward(self, inp):
        return self.model(inp)


class UnetGenerator(nn.Module):
    def __init__(self, in_channels, out_channels, nb_down_G, nb_feat_init_G, use_dropout, use_tanh):
        super(UnetGenerator, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nb_down_G = nb_down_G
        self.nb_feat_init_G = nb_feat_init_G
        self.use_dropout = use_dropout
        self.use_tanh = use_tanh
        self.build()

    def build(self):
        nb_feat_after = self.nb_feat_init_G
        self.block_down_0 = nn.Conv2d(self.in_channels, nb_feat_after, kernel_size=4, stride=2, padding=1, bias=True)
        block_up_0 = [nn.ConvTranspose2d(nb_feat_after*2, self.out_channels, kernel_size=4, stride=2, padding=1, bias=True)]
        if self.use_tanh == True :
            block_up_0 += [nn.Tanh()]
        self.block_up_0 = nn.Sequential(*block_up_0)
        for i in range(self.nb_down_G - 2):
            nb_feat_before = nb_feat_after
            nb_feat_after = min(nb_feat_after*2, 512)
            setattr(self, 'block_down_%d'%(i+1), UnetDown(nb_feat_before, nb_feat_after))
            use_dropout = True if i < 2 else False
            setattr(self, 'block_up_%d'%(i+1), UnetUp(nb_feat_after*2, nb_feat_before, use_dropout))
        nb_feat_before = nb_feat_after
        nb_feat_after = min(nb_feat_after*2, 512)
        self.block_center = UnetCenter(nb_feat_before, nb_feat_after)

    def forward(self, inp):

#        down_0 = self.block_down_0(inp)
#        down_1 = self.block_down_1(down_0)
#        down_2 = self.block_down_2(down_1)
#        down_3 = self.block_down_3(down_2)

#        center = self.block_center(down_3)

#        up_3 = self.block_up_3(torch.cat([center, down_3], 1))
#        up_2 = self.block_up_2(torch.cat([up_3, down_2], 1))
#        up_1 = self.block_up_1(torch.cat([up_2, down_1], 1))
#        up_0 = self.block_up_0(torch.cat([up_1, down_0], 1))

#        return up_0

        layers = [inp]
        for i in range(self.nb_down_G-1):
            layers.append(getattr(self, 'block_down_%d'%(i)) (layers[-1]))

        last = self.block_center(layers[-1])

        for j in range(self.nb_down_G -1):
            tmp = torch.cat([last, layers[-j-1]], 1)
            layer = getattr(self, 'block_up_%d'%(self.nb_down_G - j - 2))
            last = layer(tmp)

        return last


def define_discriminator(opt, state_dict=None, device=None):
    in_channels = opt.in_channels
    out_channels = opt.out_channels
    nb_layers = opt.nb_layers_D
    nb_feat_init = opt.nb_feat_init_D
    nb_feat_max = opt.nb_feat_max_D
    use_sigmoid = opt.use_sigmoid_D

    if nb_layers < 0 :
        raise ValueError("nb_layers must be greater than 0")
    elif nb_layers == 0 :
        discriminator = PixelDiscriminator(in_channels, out_channels, nb_feat_init, use_sigmoid)
    else :
        discriminator = PatchDiscriminator(in_channels, out_channels, nb_layers, nb_feat_init, nb_feat_max, use_sigmoid)

    if state_dict is not None :
        discriminator.load_state_dict(state_dict)
    else :
        discriminator = init_network(discriminator, init_type='normal', init_gain=0.02)

    if device is not None :
        discriminator.to(device)

    return discriminator


def define_generator(opt, state_dict=None, device=None):
    in_channels = opt.in_channels
    out_channels = opt.out_channels
    nb_down_G = opt.nb_down_G
    nb_feat_init_G = opt.nb_feat_init_G
    use_dropout = opt.use_dropout_G
    use_tanh = opt.use_tanh_G

    generator = UnetGenerator(in_channels, out_channels, nb_down_G, nb_feat_init_G, use_dropout, use_tanh)

    if state_dict is not None :
        generator.load_state_dict(state_dict)
    else :
        generator = init_network(generator, init_type='normal', init_gain=0.02)

    if device is not None :
        generator.to(device)

    return generator


def init_network(network, init_type='normal', init_gain=0.02):
    """
    네트워크의 가중치를 초기화하는 함수
    Args:
        net : nn.Module
            네트워크
        init_type : str, default='normal'
            초기화 방법
        init_gain : float, default=0.02
            초기화 gain
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1: 
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    network.apply(init_func)
    return network
