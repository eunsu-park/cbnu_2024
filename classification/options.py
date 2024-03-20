import argparse

class BaseOption():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--seed', type=int, default=1111)

    def parse(self):
        return self.parser.parse_args(args=[])

class TrainOption(BaseOption):
    def __init__(self):
        super(TrainOption, self).__init__()
        self.parser.add_argument('--is_train', type=bool, default=True)


class TestOption(BaseOption):
    def __init__(self):
        super(TestOption, self).__init__()
        self.parser.add_argument('--is_train', type=bool, default=False)
