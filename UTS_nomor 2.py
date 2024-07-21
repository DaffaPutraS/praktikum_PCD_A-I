import numpy as np
import matplotlib.pyplot as plt

# Data intensitas dan jumlah pixel
intensitas = [0, 1, 2, 3, 4, 5, 6, 7]
jumlah_pixel = [400, 600, 750, 450, 800, 0, 0, 0]

# Menampilkan histogram
plt.bar(intensitas, jumlah_pixel)
plt.title('Histogram')
plt.xlabel('Intensitas')
plt.ylabel('Jumlah Pixel')
plt.show()

# Kontras stretching (Histogram equalization)
total_pixel = sum(jumlah_pixel)
cdf = np.cumsum(jumlah_pixel) / total_pixel
equalized_intensity = [int(round(cdf[i] * 7)) for i in range(8)]

# Menampilkan hasil kontras stretching
plt.plot(intensitas, equalized_intensity, marker='o')
plt.title('Histogram Equalization')
plt.xlabel('Intensitas Asli')
plt.ylabel('Intensitas Baru')
plt.xticks(intensitas)
plt.yticks(range(8))
plt.grid(True)
plt.show()