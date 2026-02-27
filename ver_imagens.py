import numpy as np
import matplotlib.pyplot as plt

img = np.load("/home/vitor/Mama-mia/dados_processados_2d/masks/DUKE_001_slice_004.npy")

print(img.shape)  # ver dimensões


plt.imshow(img, cmap="gray")
plt.colorbar()
plt.show()