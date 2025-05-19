import cv2
import numpy as np

import matplotlib.pyplot as plt

# Carregar imagem em escala de cinza
img_path = './images/mazda_rx7.jpg'

def read_image(image_path):
    """
    Ler uma imagem em escala de cinza.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Imagem não encontrada: {image_path}")
    return image

def threshold_mean(image_path, block_size, C):
    """
    Aplicar thresholding adaptativo usando a média local.
    """
    image = read_image(image_path)
    return cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, C
    )

def threshold_gaussian(image, block_size, C):
    """
    Aplicar thresholding adaptativo usando a média ponderada gaussiana local.
    """
    image = read_image(image)
    return cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C
    )

def region_growing(image_path, threshold=10):
    """
    Region growing otimizado: compara com a média incremental da região e usa conectividade 8.
    """
    image = read_image(image_path)
    seed = (image.shape[0] // 2, image.shape[1] // 2)
    height, width = image.shape
    region = np.zeros_like(image, dtype=np.uint8)
    visited = np.zeros_like(image, dtype=bool)
    stack = [seed]
    region_sum = int(image[seed])
    region_count = 1
    region[seed] = 255
    visited[seed] = True

    # 8 vizinhos
    neighbors = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]

    while stack:
        y, x = stack.pop()
        region_mean = region_sum / region_count
        for dy, dx in neighbors:
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width and not visited[ny, nx]:
                if abs(int(image[ny, nx]) - region_mean) <= threshold:
                    region[ny, nx] = 255
                    stack.append((ny, nx))
                    region_sum += int(image[ny, nx])
                    region_count += 1
                visited[ny, nx] = True
    return region

original_image = read_image(img_path)

# Aplicar thresholding adaptativo
thresh_mean = threshold_mean(
    img_path, block_size=11, C=2
)  # Tamanho do bloco 11x11, subtrai 2 da média local

thresh_gauss = threshold_gaussian(
    img_path, block_size=11, C=2
)  # Tamanho do bloco 11x11, subtrai 2 da média ponderada gaussiana local

region_grown = region_growing(img_path, threshold=40)

# Exibir resultados
plt.figure(figsize=(10, 4))
plt.subplot(1, 4, 1)
plt.title('Original')
plt.imshow(original_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.title('Mean Threshold')
plt.imshow(thresh_mean, cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.title('Gaussian Threshold')
plt.imshow(thresh_gauss, cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.title('Region Growing')
plt.imshow(region_grown, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()