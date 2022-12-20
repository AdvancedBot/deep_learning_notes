import numpy as np
a = np.array([[56.0, 0.0, 4.4, 68.0], [1.2, 104.0, 52.0, 8.0], [1.8, 135.0, 99.0, 0.9]])
c = np.array([1,2,3,4])
print(a*c)
print(a/c)
b = np.zeros(a.shape[1])
b = a[0,:]/np.sum(a,axis=0)
# print(b)


