import random
import datetime

class Board:
    def __init__(self, size, board=[]):
        self.board = board

        if self.board == []:
            self.board = [random.randint(0, size-1) for i in range(size)]

        self.size = size

    def show(self):
        b = [self.size - 1 - i for i in self.board]

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
    
    def get_max_fitness(self):
        return (self.size-1)*(self.size)/2

    def get_fitness(self):
        horizontal_collisions = sum([self.board.count(queen)-1 for queen in self.board])/2
        diagonal_collisions = 0

        n = len(self.board)
        left_diagonal = [0] * 2*n
        right_diagonal = [0] * 2*n
        for i in range(n):
            left_diagonal[i + self.board[i] - 1] += 1
            right_diagonal[len(self.board) - i + self.board[i] - 2] += 1

        diagonal_collisions = 0
        for i in range(2*n-1):
            counter = 0
            if left_diagonal[i] > 1:
                counter += left_diagonal[i]-1
            if right_diagonal[i] > 1:
                counter += right_diagonal[i]-1
            diagonal_collisions += counter
        
        return int(self.get_max_fitness() - (horizontal_collisions + diagonal_collisions))


if __name__ == '__main__':
    new_board = [5, 3, 6, 0, 7, 1, 4, 2]
    new_board2 = [2, 0, 6, 4, 7, 1, 3, 5]
    board = Board(len(new_board2), board=new_board2)
    board.show()
    print(board.get_fitness())