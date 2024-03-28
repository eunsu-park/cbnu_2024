import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
from imageio import imread
from skimage.transform import resize
import numpy as np


class LoadData:
    """
    이미지 파일을 불러오고, 입/출력 데이터를 나누는 클래스
    """
    def __call__(self, filepath):
        """
        이미지 파일을 불러오고, 입/출력 데이터를 나누는 함수
        Args:
            filepath : str
                이미지 파일 경로
        Returns:
            inp : numpy.ndarray
                입력 데이터
            tar : numpy.ndarray
                출력 데이터
        """
        image = imread(filepath).astype(np.float32)
        inp = image[:, :1024]
        tar = image[:, 1024:]
        return inp, tar


class NormalizeData:
    """
    이미지 데이터를 -1~1 사이로 정규화하는 클래스
    """
    def __call__(self, image) :
        """
        이미지 데이터를 -1~1 사이로 정규화하는 함수
        Args:
            image : numpy.ndarray
                이미지 데이터
        Returns:
            image : numpy.ndarray
                정규화된 이미지 데이터
        """
        return image / 127.5 - 1


class ResizeData:
    """
    이미지 크기를 조절하는 클래스
    """
    def __init__(self, image_size=256):
        """
        ResizeData 클래스의 생성자
        Args:
            image_size : int or tuple
                이미지 크기
        """
        if isinstance(image_size, int) :
            image_size = (image_size, image_size)
        self.image_size = image_size

    def __call__(self, image):
        """
        이미지 크기를 조절하는 함수
        Args:
            image : numpy.ndarray
                이미지 데이터
        Returns:
            image : numpy.ndarray
                크기가 조절된 이미지 데이터
        """
        image = resize(image, (self.image_size, self.image_size), order=1, mode="constant", preserve_range=True)
        if len(image.shape) == 2 :
            image = np.expand_dims(image, axis=0)
        return image


class ToTensor:
    """
    numpy array를 torch tensor로 변환하는 클래스
    """
    def __init__(self, dtype=torch.float32):
        """
        ToTensor 클래스의 생성자
        Args:
            dtype : torch.dtype
                torch tensor의 데이터 타입
        """
        self.dtype=dtype
    def __call__(self, data):
        """
        numpy array를 torch tensor로 변환하는 함수
        Args:
            data : numpy.ndarray
                데이터
        Returns:
            data : torch.Tensor
                변환된 데이터
        """
        data = torch.tensor(data, dtype=self.dtype)
        return data
    
class CustomDataset(Dataset):
    """
    AIA-HMI Pair 데이터셋을 불러오는 클래스
    """
    def __init__(self, data_root, image_size, is_train=True):
        """
        CustomDataset 클래스의 생성자
        Args:
            data_root : str
                데이터셋이 저장된 디렉토리 경로
            is_train : bool
                학습 데이터인지 테스트 데이터인지 구분하는 플래그
        """
        if is_train is True :
            pattern = f"{data_root}/Train/*.png"
        else :
            pattern = f"{data_root}/Test/*.png"
        self.list_data = glob.glob(pattern)
        self.nb_data = len(self.list_data)
        self.transform = Compose([NormalizeData(),
                                  ResizeData(image_size),
                                  ToTensor()])

    def __len__(self):
        """
        데이터셋의 크기를 반환하는 함수
        Returns:
            int: 데이터셋의 크기
        """
        return self.nb_data
    
    def __getitem__(self, idx):
        """
        데이터셋의 idx번째 데이터를 반환하는 함수
        Args:
            idx : int
                데이터의 인덱스
        Returns:
            inp : torch.Tensor
                입력 데이터
            tar : torch.Tensor
                출력 데이터
        """
        inp, tar = LoadData()(self.list_data[idx])
        inp = self.transform(inp)
        tar = self.transform(tar)
        return inp, tar
    
def define_dataset(opt):
    """
    dataset과 dataloader를 정의하는 함수
    Args:
        opt : argparse.ArgumentParser
            사용자 입력값
    Returns:
        dataset : CustomDataset
            데이터셋 객체
        dataloader : DataLoader
            데이터로더 객체
    """
    dataset = CustomDataset(opt.data_root, is_train=True)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size,
                            shuffle=opt.is_train, num_workers=opt.num_workers)
    return dataset, dataloader
