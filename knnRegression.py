import numpy as np
import matplotlib.pyplot as plt

X = np.sort(np.random.rand(40, 1))#Feature
y = np.sin(X).ravel() #target
y[::5] += 1*(0.5-np.random.rand(8)) #gürültü ekledik
plt.scatter(X, y)
plt.show()