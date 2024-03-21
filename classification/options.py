import argparse

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--name', type=str, default='default')
        self.parser.add_argument('--seed', type=int, default=1111)
        self.parser.add_argument('--in_channels', type=int, default=1)
        self.parser.add_argument('--num_classes', type=int, default=2)
        self.parser.add_argument('--image_size', type=int, default=224)
        self.parser.add_argument("--data_root", type=str, default="path/to/data")
        self.parser.add_argument("--save_root", type=str, default="path/to/save")

    def parse(self):
        return self.parser.parse_args(args=[])


class TrainOptions(BaseOptions):
    def __init__(self):
        super(TrainOptions, self).__init__()
        self.parser.add_argument('--is_train', type=bool, default=True)
        self.parser.add_argument('--gpu_id', type=int, default=0)
        self.parser.add_argument('--batch_size', type=int, default=4)
        self.parser.add_argument('--num_workers', type=int, default=2)
        self.parser.add_argument('--num_epochs', type=int, default=10)
        self.parser.add_argument('--lr', type=float, default=0.0002)


class TestOptions(BaseOptions):
    def __init__(self):
        super(TestOptions, self).__init__()
        self.parser.add_argument('--is_train', type=bool, default=False)
        self.parser.add_argument('--gpu_id', type=int, default=0)
        self.parser.add_argument('--batch_size', type=int, default=4)
        self.parser.add_argument('--num_workers', type=int, default=2)
        self.parser.add_argument('--epoch_test', type=int, default=1)
