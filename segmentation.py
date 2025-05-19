import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_image(image_path):
    """
    Lê uma imagem em escala de cinza.

    Parâmetros:
        image_path (str): Caminho para o arquivo de imagem.

    Retorna:
        image (np.ndarray): Imagem carregada em escala de cinza.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Imagem não encontrada: {image_path}")
    
    return image

def apply_sobel(image_path, dx=1, dy=0, ksize=3):
    """
    Aplica o filtro de Sobel para detecção de bordas.

    Parâmetros:
        image_path (str): Caminho para o arquivo de imagem.
        dx (int): Ordem da derivada em x (1 para detectar bordas horizontais).
        dy (int): Ordem da derivada em y (1 para detectar bordas verticais).
        ksize (int): Tamanho do kernel do filtro Sobel (deve ser ímpar).

    Retorna:
        sobelx (np.ndarray): Resultado do filtro Sobel na direção x.
        sobely (np.ndarray): Resultado do filtro Sobel na direção y.
        sobel_magnitude (np.ndarray): Magnitude combinada das bordas.
    """
    imagem = read_image(image_path)
    sobelx = cv2.Sobel(imagem, cv2.CV_64F, dx, 0, ksize=ksize)
    sobely = cv2.Sobel(imagem, cv2.CV_64F, 0, dy, ksize=ksize)
    sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
    sobel_magnitude = np.uint8(np.clip(sobel_magnitude, 0, 255))

    return sobelx, sobely, sobel_magnitude

def apply_prewitt(image_path, ddepth=-1):
    """
    Aplica o filtro de Prewitt para detecção de bordas.

    Parâmetros:
        image_path (str): Caminho para o arquivo de imagem.
        ddepth (int): Profundidade desejada da imagem de saída (-1 para mesma profundidade da entrada).

    Retorna:
        prewittx (np.ndarray): Resultado do filtro Prewitt na direção x.
        prewitty (np.ndarray): Resultado do filtro Prewitt na direção y.
        prewitt_magnitude (np.ndarray): Magnitude combinada das bordas.
    """
    imagem = read_image(image_path)
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

def apply_canny(image_path, limiar1=100, limiar2=200, apertureSize=3, L2gradient=False):
    """
    Aplica o detector de bordas de Canny.

    Parâmetros:
        image_path (str): Caminho para o arquivo de imagem.
        limiar1 (int): Primeiro valor de threshold para o hysteresis.
        limiar2 (int): Segundo valor de threshold para o hysteresis.
        apertureSize (int): Tamanho do kernel da derivada de Sobel interna (deve ser ímpar).
        L2gradient (bool): Se True, usa a equação L2 para gradiente (mais preciso).

    Retorna:
        edges (np.ndarray): Imagem binária com as bordas detectadas.
    """
    imagem = read_image(image_path)

    return cv2.Canny(imagem, limiar1, limiar2, apertureSize=apertureSize, L2gradient=L2gradient)

# Caminho da imagem
img_path = './images/mazda_rx7.jpg'

# Aplicar filtros
sobelx, sobely, sobel_magnitude = apply_sobel(img_path, dx=1, dy=1, ksize=3)
prewittx, prewitty, prewitt_magnitude = apply_prewitt(img_path, ddepth=-1)
canny_edges = apply_canny(img_path, limiar1=100, limiar2=200, apertureSize=3, L2gradient=False)

# Exibir resultados
plt.figure(figsize=(16, 8))

plt.subplot(2, 4, 1)
plt.title('Original')
plt.imshow(read_image(img_path), cmap='gray')
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
