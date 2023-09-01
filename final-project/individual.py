import random
import math
import numpy as np

class Individual:
    def __init__(self, size=(3,3)):
        self.size = size;
        
        self.image = self.generate_cubes()
        
    def generate_cubes(self):
        #return [np.random.randint(0, 6, size=(3,3)) for _ in range(self.size[0] * self.size[1])]
        rows = []
        
        for row_index in range(self.size[0]):
            grouped_rows = [[] for _ in range(3)]
            
            for col_index in range(self.size[1]):
                cube = np.random.randint(0, 6, size=(3, 3))
                
                for i, cube_row in enumerate(cube):
                    grouped_rows[i].extend(cube_row)
            
            rows.extend(grouped_rows)
            
        return np.vstack(rows)
    
    def display(self):
        print(self.image)
        
