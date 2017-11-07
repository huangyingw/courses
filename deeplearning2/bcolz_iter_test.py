
# coding: utf-8

from bcolz_array_iterator2 import BcolzArrayIterator2


from bcolz import carray


import numpy as np


x = np.arange(14)
x


y = np.arange(14)
y


x = carray(x, chunklen=3)
y = carray(y, chunklen=3)


b = BcolzArrayIterator2(x, y, shuffle=True, batch_size=3)


b.N


nit = len(x) // b.batch_size + 1
nit


for j in range(10000):
    bx, by = list(zip(*[next(b) for i in range(nit)]))
    nx = np.concatenate(bx)
    ny = np.concatenate(by)
    assert(np.allclose(nx, ny))
    assert(len(np.unique(nx)) == len(nx))


[next(b) for i in range(20)]
