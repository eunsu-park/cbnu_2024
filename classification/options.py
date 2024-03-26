import argparse

class BaseOptions():
    """
    BaseOptions 클래스
    모델 개발 전반에 걸쳐 사용되는 옵션을 정의하는 클래스
    """
    def __init__(self):
        """
        BaseOptions 클래스의 생성자
        argparse.ArgumentParser 객체를 생성하고, 기본적인 옵션을 추가한다.
        """
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--name', type=str, default='default')
        self.parser.add_argument('--seed', type=int, default=1111)
        self.parser.add_argument('--in_channels', type=int, default=1)
        self.parser.add_argument('--num_classes', type=int, default=2)
        self.parser.add_argument('--image_size', type=int, default=224)
        self.parser.add_argument("--data_root", type=str, default="/Users/eunsu/CBNU/Classification/Dataset")
        self.parser.add_argument("--save_root", type=str, default="/Users/eunsu/CBNU/Classification/Result")
        # self.parser.add_argument("--data_root", type=str, default="path/to/data")
        # self.parser.add_argument("--save_root", type=str, default="path/to/save")


    def parse(self):
        """
        입력된 인자를 파싱하는 함수
        Returns:
            argparse.Namespace: 파싱된 인자
        """
        return self.parser.parse_args(args=[])


class TrainOptions(BaseOptions):
    """
    TrainOptions 클래스
    학습에 필요한 옵션을 정의하는 클래스
    """
    def __init__(self):
        """
        TrainOptions 클래스의 생성자
        BaseOptions 클래스의 생성자를 호출하고, 추가적인 옵션을 정의한다.
        """
        super(TrainOptions, self).__init__()
        self.parser.add_argument('--is_train', type=bool, default=True)
        self.parser.add_argument('--gpu_id', type=int, default=0)
        self.parser.add_argument('--batch_size', type=int, default=4)
        self.parser.add_argument('--num_workers', type=int, default=2)
        self.parser.add_argument('--num_epochs', type=int, default=10)
        self.parser.add_argument('--lr', type=float, default=0.0002)


class TestOptions(BaseOptions):
    """
    TestOptions 클래스
    테스트에 필요한 옵션을 정의하는 클래스
    """
    def __init__(self):
        """
        TestOptions 클래스의 생성자
        BaseOptions 클래스의 생성자를 호출하고, 추가적인 옵션을 정의한다.
        """
        super(TestOptions, self).__init__()
        self.parser.add_argument('--is_train', type=bool, default=False)
        self.parser.add_argument('--gpu_id', type=int, default=0)
        self.parser.add_argument('--batch_size', type=int, default=4)
        self.parser.add_argument('--num_workers', type=int, default=2)
        self.parser.add_argument('--epoch_test', type=int, default=1)
