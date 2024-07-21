import matplotlib.pyplot as plt

# Data intensitas derajat keabuan
intensitas = [0, 1, 2, 3, 4, 5, 6, 7]
jumlah_piksel = [400, 600, 750, 450, 800, 0, 0, 0]

# Gambar histogram
plt.bar(intensitas, jumlah_piksel)
plt.xlabel('Intensitas')
plt.ylabel('Jumlah Piksel')
plt.title('Histogram citra f(x) UTS')
plt.show()
