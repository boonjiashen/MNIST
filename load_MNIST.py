"""Load MNIST and display random digits
"""

import sklearn.datasets
import matplotlib.pyplot as plt
import random
import util

# Load MNIST digits
data = sklearn.datasets.fetch_mldata('MNIST original')
X = data.data  # (n_examples, n_features)
n_examples, n_features = X.shape

# Sample from digits
sample = X[random.sample(range(n_examples), 100), :]

# Display sample
sample = sample.reshape(-1, 28, 28)
canvas = util.tile(sample)
plt.imshow(canvas)
plt.show()
