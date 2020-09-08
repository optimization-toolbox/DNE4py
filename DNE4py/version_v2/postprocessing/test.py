import numpy as np


a = np.array([[12341], [5123, 123], [1234]], dtype=object)
b = np.array([[5123], [1234], [1237, 4124, 1231]], dtype=object)

print()
print(a)
print(b)
print()

with open('test.npy', 'ab') as f:
    np.save(f, a)

with open('test.npy', 'rb') as f:
    for _ in range(1):
        print(np.load(f, allow_pickle=True))
print()

with open('test.npy', 'ab') as f:
    np.save(f, b)

with open('test.npy', 'rb') as f:
    print('!!!')
    for _ in range(2):
        print(np.load(f, allow_pickle=True))
print()
