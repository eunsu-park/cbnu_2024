import numpy as np

# x.dtype : 데이터 타입 확인

x = np.array([1,2,3,4,5])
print(x.dtype)

x = np.array([1.,2.,3.,4.,5.])
print(x.dtype)

x = np.array([1.,2.,3.,4.,5.],
             dtype=np.float32)
print(x.dtype)
