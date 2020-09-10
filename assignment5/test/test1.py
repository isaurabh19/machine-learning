from source.utils import SquareConcept
from joblib import Parallel, delayed
import numpy as np
import random


def test1():
    concept1 = SquareConcept((-4., 3.), (2., 1.), (-2., -1.), 3)
    concept2 = SquareConcept((-4., 3.), (2., 0), (-1., -2.), 1)

    assert concept1.return_label((-5, 0)) == -1
    assert concept1.return_label((-1, -3)) == 1
    assert concept1.return_label((3, -3)) == -1
    assert concept1.return_label((1, 0)) == -1
    assert concept1.return_label((5, 3)) == -1


# test1()
import source.utils as utils
import matplotlib.pyplot as plt

# data = utils._get_labelled_data(1)
# x1 = data[:, :1]
# x2 = data[:, 1:-1]
step = 0.1
xx, yy = np.meshgrid(np.arange(-6, 7 + step, step), np.arange(-4, 5 + step, step))
grid = np.c_[xx.ravel(), yy.ravel()]
concept = SquareConcept((-4., 3.), (2., 1.), (-2., -1.), 3)
labels = np.array(list(map(concept.return_label, grid)))
data = np.concatenate((grid, np.array([labels]).T), axis=1)

X = data[:, :-1]
y = data[:, -1:]
z = y.reshape(xx.shape)
z = np.flipud(z)
plt.imshow(z, extent=[xx.min(), xx.max(), yy.min(), yy.max()])
plt.show()
