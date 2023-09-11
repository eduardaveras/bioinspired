import cv2
import numpy as np
import time

# Carrega a imagem a partir de um path
def load_image(path):
    image = cv2.imread(path)
    # Cut to a square
    image = image[:max(image.shape), :max(image.shape)]
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
def image_to_gray_pixels(image, size, pixel_size, alpha=1.5, beta=30):
    # Processo de filtragem da imagem
    image = cv2.resize(image, (size, size))
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_equalized = equalize_image(image_hsv, alpha=alpha, beta=beta)

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image_gray = np.zeros((size//pixel_size, size//pixel_size), dtype=np.uint8)

    for i in range(0, size, pixel_size):
        for j in range(0, size, pixel_size):
            cube = grayscale[i:i+pixel_size, j:j+pixel_size]
            # Calcula a luminância média do cubo
            mean_luminance = np.mean(cube)
            # Atribui aquele cubo da imagem a essa luminância média
            image_gray[i//pixel_size, j//pixel_size] = mean_luminance


    return image_gray


# Da imagem retorna os cubos (3x3) em um array
# Isso deve ser feito depois da imagem ser pixelada em tons de cinza
def gray_image_to_cubes(image):
    if image.shape[0] % 3:
        raise ValueError("Image size must be divisible by 3")

    cubes = []

    for i in range(0, image.shape[0], 3):
        row_cubes = []
        for j in range(0, image.shape[1], 3):
            cube = image[i:i+3, j:j+3]
            row_cubes.append(cube)
        cubes.append(row_cubes)

    return np.array(cubes)

def cubes_to_image(cubes, stroke_width=0, size_times=1, stroke_color=255):
    image = np.zeros((cubes.shape[0]*3*size_times, cubes.shape[1]*3*size_times), dtype=np.uint8)

    for i in range(0, image.shape[0], 3*size_times):
        for j in range(0, image.shape[1], 3*size_times):
            cube = cubes[i//(3*size_times), j//(3*size_times)]
            cube_times = np.kron(cube, np.ones((size_times, size_times), dtype=np.uint8))

            # Border in cube_times
            if stroke_width > 0:
                cube_times[0:stroke_width, :] = stroke_color
                cube_times[-stroke_width:, :] = stroke_color
                cube_times[:, 0:stroke_width] = stroke_color
                cube_times[:, -stroke_width:] = stroke_color


            image[i:i+3*size_times, j:j+3*size_times] = cube_times
            # image[i_:i_+3*size_times, j_:j_+3*size_times] = cube_times

    return image

# Retorna a imagem pixelada em tamanho size x size
def resize_pixeled(image, size):
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)

# Coloca a imagem e um arquivo
def save_image(image, path):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(path, image_hsv)

# Para mostrar a imagem no jupyter!
from PIL import Image
from IPython.display import display

def display_image(image):
    display(Image.fromarray(image))
