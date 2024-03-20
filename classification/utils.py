import random
import torch
import numpy as np

def fix_seed(seed):
    """
    seed를 고정하는 함수

    Args:
        seed : int
    Returns:
        None
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_num_params(model, only_grad=False):
    """
    모델(레이어)의 파라미터 수를 계산하는 함수
    
    Args:
        model : torch.nn.Module
        only_grad : bool, default=False
            True인 경우, requires_grad=True인 파라미터만 계산
    Returns:
        num_params : int
    """
    if only_grad:
        return sum([p.numel() for p in model.parameters() if p.requires_grad])
    else :
        return sum([p.numel() for p in model.parameters()])
