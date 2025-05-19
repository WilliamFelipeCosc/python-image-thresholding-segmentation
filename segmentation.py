import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_sobel(imagem, dx=1, dy=0, ksize=3):
    sobelx = cv2.Sobel(imagem, cv2.CV_64F, dx, 0, ksize=ksize)
    sobely = cv2.Sobel(imagem, cv2.CV_64F, 0, dy, ksize=ksize)
    sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
    sobel_magnitude = np.uint8(np.clip(sobel_magnitude, 0, 255))
    
    return sobelx, sobely, sobel_magnitude

def apply_prewitt(imagem, ddepth=-1):
    kernel_prewitt_x = np.array([[1, 0, -1],
                                 [1, 0, -1],
                                 [1, 0, -1]], dtype=np.float32)
    kernel_prewitt_y = np.array([[1, 1, 1],
                                 [0, 0, 0],
                                 [-1, -1, -1]], dtype=np.float32)
    prewittx = cv2.filter2D(imagem, ddepth, kernel_prewitt_x)
    prewitty = cv2.filter2D(imagem, ddepth, kernel_prewitt_y)
    prewitt_magnitude = np.sqrt(prewittx.astype(np.float32)**2 + prewitty.astype(np.float32)**2)
    prewitt_magnitude = np.uint8(np.clip(prewitt_magnitude, 0, 255))
    return prewittx, prewitty, prewitt_magnitude

def apply_canny(imagem, limiar1=100, limiar2=200, apertureSize=3, L2gradient=False):
    return cv2.Canny(imagem, limiar1, limiar2, apertureSize=apertureSize, L2gradient=L2gradient)

# Carregar imagem em escala de cinza
imagem = cv2.imread('./images/mazda_rx7.jpg', cv2.IMREAD_GRAYSCALE)

# Aplicar filtros
sobelx, sobely, sobel_magnitude = apply_sobel(imagem, dx=1, dy=1, ksize=3)
prewittx, prewitty, prewitt_magnitude = apply_prewitt(imagem, ddepth=-1)
canny_edges = apply_canny(imagem, limiar1=100, limiar2=200, apertureSize=3, L2gradient=False)

# Exibir resultados
plt.figure(figsize=(16, 8))

plt.subplot(2, 4, 1)
plt.title('Original')
plt.imshow(imagem, cmap='gray')
plt.axis('off')

plt.subplot(2, 4, 2)
plt.title('Sobel X')
plt.imshow(np.abs(sobelx), cmap='gray')
plt.axis('off')

plt.subplot(2, 4, 3)
plt.title('Sobel Y')
plt.imshow(np.abs(sobely), cmap='gray')
plt.axis('off')

plt.subplot(2, 4, 4)
plt.title('Sobel Magnitude')
plt.imshow(sobel_magnitude, cmap='gray')
plt.axis('off')

plt.subplot(2, 4, 5)
plt.title('Canny')
plt.imshow(canny_edges, cmap='gray')
plt.axis('off')

plt.subplot(2, 4, 6)
plt.title('Prewitt X')
plt.imshow(prewittx, cmap='gray')
plt.axis('off')

plt.subplot(2, 4, 7)
plt.title('Prewitt Y')
plt.imshow(prewitty, cmap='gray')
plt.axis('off')

plt.subplot(2, 4, 8)
plt.title('Prewitt Magnitude')
plt.imshow(prewitt_magnitude, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
