import numpy as np
from matplotlib import pyplot as plt
import pickle

f = open ('outfile', 'rb')
data = pickle.load(f)
assert all([d in [0,100] for d in data])
mat = np.reshape(data,(-1,356))
map_obstacles = np.argwhere(mat>0)
resolution = 0.03
map_obstacles = map_obstacles*resolution
map_obstacles = np.fliplr(map_obstacles)
map_obstacles[:,0] += -1.94
map_obstacles[:,1] += -8.63
print(map_obstacles)
'''plot.imshow(mat)
plt.gca().invert_yaxis()
plt.show()'''
plt.scatter(map_obstacles[:,0],map_obstacles[:,1],s=0.5)
plt.show()
