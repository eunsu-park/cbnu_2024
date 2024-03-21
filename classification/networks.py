import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(MyModel, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.define_classifier()
        self.define_feature_extractor()

    def define_feature_extractor(self):
        model1 = []
        model1 += [nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1)]
        model1 += [nn.BatchNorm2d(32)]
        model1 += [nn.ReLU()]
        model1 += [nn.MaxPool2d(kernel_size=2, stride=2)]
        model1 += [nn.Conv2d(32, 64, kernel_size=3, padding=1)]
        model1 += [nn.BatchNorm2d(64)]
        model1 += [nn.ReLU()]
        model1 += [nn.MaxPool2d(kernel_size=2, stride=2)]
        self.feature_extractor = nn.Sequential(*model1)

    def define_classifier(self):
        model2 = []
        model2 += [nn.Linear(4096, 4096)]
        model2 += [nn.ReLU()]
        model2 += [nn.Dropout(p=0.5)]
        model2 += [nn.Linear(4096, self.num_classes)]
        model2 += [nn.Sigmoid()]
        self.classifier = nn.Sequential(*model2)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)   
        x = self.classifier(x)
