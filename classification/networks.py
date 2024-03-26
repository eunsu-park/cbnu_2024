import torch
import torch.nn as nn

class CustomModel(nn.Module):
    """
    CustomModel 클래스
    """
    def __init__(self, in_channels, num_classes):
        """
        CustomModel 클래스의 생성자
        Args:
            in_channels : int
                입력 이미지의 채널 수
            num_classes : int
                클래스 수
        """
        super(CustomModel, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.define_classifier()
        self.define_feature_extractor()

    def define_feature_extractor(self):
        """
        feature extractor를 정의하는 함수
        """
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
        """
        classifier를 정의하는 함수
        """
        model = []

        model += [nn.Linear(512*7*7, 4096),
                  nn.ReLU(inplace=True),
                  nn.Dropout()]
        model += [nn.Linear(4096, 4096),
                  nn.ReLU(inplace=True),
                  nn.Dropout()]
        model += [nn.Linear(4096, self.num_classes),
                  nn.Softmax(dim=1)]

        self.classifier = nn.Sequential(*model)

    def forward(self, x):
        """
        forward propagation 함수
        Args:
            x : torch.Tensor
                입력 데이터
        Returns:
            x : torch.Tensor
                출력 데이터
        """
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)   
        x = self.classifier(x)
        return x


def define_network(opt):
    """
    네트워크를 정의하는 함수
    Args:
        opt : argparse.Namespace
            옵션 객체
    Returns:
        network : nn.Module
            네트워크
    """
    network = CustomModel(opt.in_channels, opt.num_classes)
    if opt.is_train:
        network.apply(weight_init)
    return network


def weight_init(m):
    """
    네트워크의 가중치를 초기화하는 함수
    Args:
        m : nn.Module
            네트워크의 각 층
    """
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        nn.init.constant_(m.bias, 0)


def define_criterion(opt):
    """
    loss 함수를 정의하는 함수
    Args:
        opt : argparse.Namespace
            옵션 객체
    Returns:
        criterion : nn.Module
            loss 함수
    """
    criterion = nn.CrossEntropyLoss()
    return criterion


def define_optimizer(network, opt):
    """
    optimizer를 정의하는 함수
    Args:
        network : nn.Module
            네트워크
        opt : argparse.Namespace
            옵션 객체
    Returns:
        optimizer : torch.optim
            optimizer
    """
    optimizer = torch.optim.Adam(network.parameters(), lr=opt.lr)
    return optimizer
