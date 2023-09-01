import cv2
import numpy as np
import time

# Carrega a imagem a partir de um path
def load_image(path):
    image = cv2.imread(path)
    # Convert to hsv
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# Apenas para ajustar os valores de luminância da imagem
def equalize_image(image, alpha=1.5, beta=30):
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    img_yuv = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2YUV)

    # Equaliza o histograma do canal Y
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

    # Convert back to BGR
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img_output


# Retorna os tons de cinza da imagem em um array size//cube_size x size//cube_size
def image_to_gray_pixels(image, size, cube_size, alpha=1.5, beta=30):
    # Processo de filtragem da imagem
    image = cv2.resize(image, (size, size))
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_equalized = equalize_image(image_hsv, alpha=alpha, beta=beta)

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image_gray = np.zeros((size//cube_size, size//cube_size), dtype=np.uint8)

    for i in range(0, size, cube_size):
        for j in range(0, size, cube_size):
            cube = grayscale[i:i+cube_size, j:j+cube_size]
            # Calcula a luminância média do cubo
            mean_luminance = np.mean(cube)
            # Atribui aquele cubo da imagem a essa luminância média
            image_gray[i//cube_size, j//cube_size] = mean_luminance

    return image_gray


# Da imagem retorna os cubos (3x3) em um array
# Isso deve ser feito depois da imagem ser pixelada
def image_cubes(image):
    cubes = []
    for i in range(0, image.shape[0], 3):
        for j in range(0, image.shape[1], 3):
            cube = image[i:i+3, j:j+3]
            cubes.append(cube)

    return cubes

# Retorna a imagem pixelada em tamanho size x size
def resize_pixeled(image, size):
    return cv2.resize(image, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA)

# Junto os cubos e retorna a imagem
def cubes_to_image(cubes, size, cube_size):
    reconstructed_image = np.zeros((size, size, 3), dtype=np.uint8)

    cube_idx = 0
    for i in range(0, size, cube_size):
        for j in range(0, size, cube_size):
            reconstructed_image[i:i+cube_size, j:j+cube_size] = cubes[cube_idx]
            cube_idx += 1

    return reconstructed_image

# Para mostrar a imagem no jupyter!
from PIL import Image
from IPython.display import display

def display_image(image):
    display(Image.fromarray(image))
