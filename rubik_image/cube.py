import numpy as np

class Cube:
    def __init__(self, _map):
        if type(_map) is not np.ndarray:
            raise TypeError("Map must be a numpy array")

        if _map.shape != (3, 3):
            raise ValueError("Map must be a 3x3 matrix")

        if len(_map) != 3 or len(_map[0]) != 3:
            raise ValueError("Map must be a 3x3 matrix")

        # mapeia de 0 a 1 onde:
        # 0 = branco
        # 1 = azul
        # 2 = verde
        # 3 = vermelho
        # 4 = amarelo 
        # 5 = laranja
        self.map = _map
        # gray_map sao os valores de luminancia de cada cubo
        # Isso vai ser utilizado para calcular o fitness
        self.gray_map = self.gray_array()

    def __str__(self):
        return str(self.map)

    def map_color(self, k, gray_mode=False):
        color = (0, 0, 0)
        color_gray = 0

        if k == 0: # Branco
            color = (255, 255, 255)
            color_gray = 255
        elif k == 1: # Azul
            color = (0, 0, 255)
            color_gray = 29
        elif k == 2: # Verde
            color = (0, 255, 0)
            color_gray = 149
        elif k == 3: # Vermelho
            color = (255, 0, 0)
            color_gray = 76
        elif k == 4: # Amarelo
            color = (255, 255, 0)
            color_gray = 178
        elif k == 5: # Laranja
            color = (255, 128, 0)
            color_gray = 125

        return color_gray if gray_mode else color

    def gray_array(self):
        cube = np.zeros((3, 3), dtype=np.uint8)

        for i in range(3):
            for j in range(3):
                cube[i][j] = self.map_color(self.map[i][j], gray_mode=True)

        return cube

    def image(self, size, stroke_width):
        image = np.zeros((size * 3, size * 3, 3), np.uint8)
        for i in range(3):
            for j in range(3):
                color = self.map[i][j]
                fill = self.map_color(color)

                # Preencher o quadrado com a cor
                image[i * size:(i + 1) * size, j * size:(j + 1) * size] = fill

                # Adicionar o "stroke" branco
                image[i * size:(i + 1) * size, j * size:j * size] = (255, 255, 255)  # Esquerda
                image[i * size:(i + 1) * size, (j + 1) * size : (j + 1) * size] = (255, 255, 255)  # Direita
                image[i * size:i * size, j * size:(j + 1) * size] = (255, 255, 255)  # Topo
                image[(i + 1) * size :(i + 1) * size, j * size:(j + 1) * size] = (255, 255, 255)  # Fundo

                # Add a grid
                if stroke_width != 0:
                    image[i * size:(i + 1) * size, j * size:j * size + stroke_width] = (255, 255, 255)  # Esquerda
                    image[i * size:(i + 1) * size, (j + 1) * size - stroke_width:(j + 1) * size] = (255, 255, 255)  # Direita
                    image[i * size:i * size + stroke_width, j * size:(j + 1) * size] = (255, 255, 255)  # Topo
                    image[(i + 1) * size - stroke_width:(i + 1) * size, j * size:(j + 1) * size] = (255, 255, 255)  # Fundo

        return image