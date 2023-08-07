import numpy as np

# Functions to be solved
class Functions:
    def __init__(self, function_name, dimensions):
        self.function = None
        self.bounds = None
        self.dimensions = dimensions
        self.name = function_name
        # self.tolerance = 1e-16

        if function_name == "ackley":
            self.function = self.ackley
            self.bounds = [-32.768, 32.768]
        elif function_name == "rastrigin":
            self.function = self.rastrigin
            self.bounds = [-5.12, 5.12]
        elif function_name == "schwefel":
            self.function = self.schwefel
            self.bounds = [-500, 500]
        elif function_name == "rosenbrock":
            self.function = self.rosenbrock
            self.bounds = [-2.048, 2.048]
        else: raise Exception("Function not found")

    def __call__(self, X):
        return self.function(X)

    def ackley(self, X):
        # X: 1D array of length d
        # d: dimension of the domain of the function

        # Return: float
        # Ackley's function
        d = self.dimensions
        value = -20.0 * np.exp(-0.2 * np.sqrt(1 / d * np.sum(X ** 2))) - np.exp(
            1 / d * np.sum(np.cos(2 * np.pi * X))) + 20 + np.exp(1)

        return value
        # return value if abs(value) > self.tolerance else 0.0

    def rastrigin(self, X):
        # X: 1D array of length d
        # d: dimension of the domain of the function

        # Return: float
        # Rastrigin's function
        d = self.dimensions
        value = 10 * d + np.sum(X ** 2 - 10 * np.cos(2 * np.pi * X))

        return value
        # return value if abs(value) > self.tolerance else 0.0

    def schwefel(self, X):
        # X: 1D array of length d
        # d: dimension of the domain of the function

        # Return: float
        # Schwefel's function
        d = self.dimensions
        value = 418.9829 * d - np.sum(X * np.sin(np.sqrt(np.abs(X))))

        return value
        # return value if abs(value) > self.tolerance else 0.0

    def rosenbrock(self, X):
        # X: 1D array of length d
        # d: dimension of the domain of the function

        # Return: float
        # Rosenbrock's function
        d = self.dimensions
        value = np.sum(100 * (X[1:] - X[:-1] ** 2) ** 2 + (1 - X[:-1]) ** 2)

        return value
        # return value if abs(value) > self.tolerance else 0.0