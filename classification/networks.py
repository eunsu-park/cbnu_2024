import torch
import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CustomModel, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.define_classifier()
        self.define_feature_extractor()

    def define_feature_extractor(self):
        model = []

        model += [nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1),
                  nn.BatchNorm2d(64), nn.ReLU()]
        model += [nn.Conv2d(64, 64, kernel_size=3, padding=1),
                  nn.BatchNorm2d(64), nn.ReLU()]
        model += [nn.MaxPool2d(kernel_size=2, stride=2)]

        model += [nn.Conv2d(64, 128, kernel_size=3, padding=1),
                  nn.BatchNorm2d(128), nn.ReLU()]
        model += [nn.Conv2d(128, 128, kernel_size=3, padding=1),
                  nn.BatchNorm2d(128), nn.ReLU()]
        model += [nn.MaxPool2d(kernel_size=2, stride=2)]

        model += [nn.Conv2d(128, 256, kernel_size=3, padding=1),
                  nn.BatchNorm2d(256), nn.ReLU()]
        model += [nn.Conv2d(256, 256, kernel_size=3, padding=1),
                  nn.BatchNorm2d(256), nn.ReLU()]
        model += [nn.Conv2d(256, 256, kernel_size=3, padding=1),
                  nn.BatchNorm2d(256), nn.ReLU()]
        model += [nn.Conv2d(256, 256, kernel_size=3, padding=1),
                  nn.BatchNorm2d(256), nn.ReLU()]
        model += [nn.MaxPool2d(kernel_size=2, stride=2)]

        model += [nn.Conv2d(256, 512, kernel_size=3, padding=1),
                  nn.BatchNorm2d(512), nn.ReLU()]
        model += [nn.Conv2d(512, 512, kernel_size=3, padding=1),
                  nn.BatchNorm2d(512), nn.ReLU()]
        model += [nn.Conv2d(512, 512, kernel_size=3, padding=1),
                  nn.BatchNorm2d(512), nn.ReLU()]
        model += [nn.Conv2d(512, 512, kernel_size=3, padding=1),
                  nn.BatchNorm2d(512), nn.ReLU()]
        model += [nn.MaxPool2d(kernel_size=2, stride=2)]

        model += [nn.Conv2d(512, 512, kernel_size=3, padding=1),
                  nn.BatchNorm2d(512), nn.ReLU()]
        model += [nn.Conv2d(512, 512, kernel_size=3, padding=1),
                  nn.BatchNorm2d(512), nn.ReLU()]
        model += [nn.Conv2d(512, 512, kernel_size=3, padding=1),
                  nn.BatchNorm2d(512), nn.ReLU()]
        model += [nn.Conv2d(512, 512, kernel_size=3, padding=1),
                  nn.BatchNorm2d(512), nn.ReLU()]
        model += [nn.MaxPool2d(kernel_size=2, stride=2)]
        
        self.feature_extractor = nn.Sequential(*model)

    def define_classifier(self):
        model = []

        model += [nn.Linear(512*7*7, 4096),
                    nn.ReLU(inplace=True),
                    nn.Dropout()]
        model += [nn.Linear(4096, 4096),
                    nn.ReLU(inplace=True),
                    nn.Dropout()]
        model += [nn.Linear(4096, self.num_classes)]
        model += [nn.Softmax(dim=1)]

        self.classifier = nn.Sequential(*model)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)   
        x = self.classifier(x)
        return x


def define_network(opt):
    model = CustomModel(opt.in_channels, opt.num_classes)
    model.apply(weight_init)
    return model


def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        nn.init.constant_(m.bias, 0)


def define_criterion(opt):
    criterion = nn.CrossEntropyLoss()
    return criterion


def define_optimizer(model, opt):
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    return optimizer
