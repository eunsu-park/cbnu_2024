# 03_dtype.py
# 데이터의 타입 확인

import numpy as np

# 정수형 데이터
x = np.array([1,2,3,4,5])
print(f"Data Type: {x.dtype}") # x.dtype : 데이터 타입 확인

# 정수형 데이터 (dtype 지정)
x= np.array([1,2,3,4,5], dtype=np.int32)
print(f"Data Type: {x.dtype}")

# 실수형 데이터
x = np.array([1.,2.,3.,4.,5.])
print(f"Data Type: {x.dtype}")

# 실수형 데이터 (dtype 지정)
x = np.array([1.,2.,3.,4.,5.], dtype=np.float32)
print(f"Data Type: {x.dtype}")
