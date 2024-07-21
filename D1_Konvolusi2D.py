import cv2
import numpy as np
from matplotlib import pyplot as plt


def add_noise(image, noise_type):
    if noise_type == "salt_pepper":
        row, col, ch = image.shape
        s_vs_p = 0.5
        amount = 0.04
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        out[coords[0], coords[1], :] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        out[coords[0], coords[1], :] = 0
        return out
    elif noise_type == "gaussian":
        row, col, ch = image.shape
        mean = 0
        var = 0.1
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        return noisy
    elif noise_type == "spike":
        row, col, ch = image.shape
        amount = 0.004
        out = np.copy(image)
        num_spike = np.ceil(amount * image.size)
        coords = [np.random.randint(0, i - 1, int(num_spike)) for i in image.shape]
        out[coords[0], coords[1], :] = 255  # Assuming white spikes
        return out
    else:
        return image


def convolve2D(image, kernel):
    i_height, i_width = image.shape[:2]
    k_height, k_width = kernel.shape[:2]
    pad_height = k_height // 2
    pad_width = k_width // 2

    # Pad the image with zeros on the border
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width), (0, 0)), mode='constant',
                          constant_values=0)

    output = np.zeros((i_height, i_width, 3))

    for i in range(pad_height, i_height + pad_height):
        for j in range(pad_width, i_width + pad_width):
            for k in range(3):  # Assuming the image has 3 channels (RGB)
                region = padded_image[i - pad_height:i + pad_height + 1, j - pad_width:j + pad_width + 1, k]
                output[i - pad_height, j - pad_width, k] = np.sum(region * kernel)

    return output


# Daftar nama file gambar
image_files = ['image/gambar_1.jpg', 'image/gambar_2.jpg', 'image/gambar_3.jpg', 'image/gambar_4.jpg', 'image/gambar_5.jpg']

# Definisi kernel (misal kernel blur)
kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9.0

plt.figure(figsize=(20, 15))

for idx, file in enumerate(image_files):
    # Membaca citra asli
    image = cv2.imread(file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Menambahkan noise pada citra
    noisy_image_sp = add_noise(image, "salt_pepper")
    noisy_image_gaussian = add_noise(image, "gaussian")
    noisy_image_spike = add_noise(image, "spike")

    # Melakukan konvolusi
    convolved_image_sp = convolve2D(noisy_image_sp, kernel)

    # Menampilkan citra asli dan citra dengan berbagai noise
    plt.subplot(len(image_files), 5, idx * 5 + 1), plt.imshow(image), plt.title(f'Original Image {idx+1}')
    plt.subplot(len(image_files), 5, idx * 5 + 2), plt.imshow(noisy_image_sp), plt.title(f'Salt & Pepper Noise {idx+1}')
    plt.subplot(len(image_files), 5, idx * 5 + 3), plt.imshow(noisy_image_gaussian), plt.title(f'Gaussian Noise {idx+1}')
    plt.subplot(len(image_files), 5, idx * 5 + 4), plt.imshow(noisy_image_spike), plt.title(f'Spike Noise {idx+1}')
    plt.subplot(len(image_files), 5, idx * 5 + 5), plt.imshow(convolved_image_sp.astype(np.uint8)), plt.title(f'Convolved Image {idx+1}')

plt.show()
