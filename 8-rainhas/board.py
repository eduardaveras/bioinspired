import random
import datetime
import math

class Board:
    def __init__(self, size, dna=[]):

        if dna == []:
            dna = [random.randrange(0, 2) for i in range(size * (len(bin(size - 1)) - 2))]

        # dna to string
        self.dna = ''.join(map(str, dna))

        self.size = size
        self.fitness = self.get_fitness()

        if self.fitness == self.get_max_fitness():
            self.isSolution = True
        else:
            self.isSolution = False

    def __str__(self):
        return self.dna

    def __len__(self):
        return self.dna

    def show(self):
        board = self.get_board()
        b = [self.size - 1 - i for i in board]

        for i in range(0, self.size):
            print(self.size - i , end=' ')
            for j in range(0, self.size):
                if b[j] == i:
                    print('Q', end=' ')
                else:
                    print('.', end=' ')
            print()

        print(' ', end=' ')
        for i in range(1, self.size + 1):
            print(i, end=' ')

        print()
        print("Fitness: ", self.fitness)

    def get_board(self):
        # transform the board list to a binary string
        binary = ''
        for i in range(self.size):
            # binary string board to list
            board = []
            s = len(bin(self.size - 1)) - 2

            for k in range(0, len(self.dna), s):
                board.append(int(self.dna[k:k+s], 2))

        return board

    def get_max_fitness(self):
        return ((self.size-1)*(self.size)/2)

    def get_fitness(self):
        board = self.get_board()
        max_fitness = (self.size-1)*(self.size)/2
        horizontal_collisions = sum([board.count(queen)-1 for queen in board])/2
        diagonal_collisions = 0

        n = len(board)
        left_diagonal = [0] * 2*n
        right_diagonal = [0] * 2*n
        for i in range(n):
            left_diagonal[i + board[i] - 1] += 1
            right_diagonal[len(board) - i + board[i] - 2] += 1

        diagonal_collisions = 0
        for i in range(2*n-1):
            counter = 0
            if left_diagonal[i] > 1:
                counter += left_diagonal[i]-1
            if right_diagonal[i] > 1:
                counter += right_diagonal[i]-1
            diagonal_collisions += counter

        return int(max_fitness - (horizontal_collisions + diagonal_collisions))


def binary_to_board(binary, size):
    # binary string board to list
    board = []
    for k in range(0, len(binary), size):
        board.append(int(binary[k:k+size], 2))

        return board


def board_to_binary(board, size):
    # transform the board list to a binary string
    binary = ''
    size = size - 1
    # 0 até size-1
    for i in range(size):
        num_bin = bin(board[i])[2:]
        bin_size = len(bin(size)) - 2
        for j in range(len(num_bin), bin_size, 1):
            binary += '0'
        binary += bin(board[i])[2:]

    return binary


def new_board(size, dna=[]):
    return Board(size, dna)


if __name__ == '__main__':
    new_board = [5, 3, 6, 0, 7, 1, 4, 2]
    new_board2 = [2, 0, 6, 4, 7, 1, 3, 5]
    new_board3 = Board(len(new_board), board=new_board)

    print(Board.boardTobinary(new_board3))
    # board = Board(len(new_board2), board=new_board2)
    # board.show()
    # print(board.gene)
    # print(board.get_fitness())