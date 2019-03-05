import numpy as np
from model import SimilarityModel

a = SimilarityModel(10, 10, 100, np.random.rand(100, 10), 1, 'cpu')
for param in a.named_parameters():
    print(param)
