from typing import Type
import numpy as np


class Perceptron():
    def __init__(self):
        self.input_length = 0
        self.indices_list = None
    
    def apply(self, x_list:list[int]):
        return 0
    
    def show_truth_table(self):
        print()

    # @classmethod
    # def check_length(cls, n, list_):
    #     if len(list_) != n:
    #         raise ValueError(f'List must have exactly {n} elements.')


class SimplePerceptron(Perceptron):
    def __init__(self, w_list:list[int], b, indices=None):
        super().__init__()
        self.w_list = w_list
        self.b = b
        self.indices = indices if indices is not None else list(range(len(self.w_list)))
        self.input_length = len(self.indices)

    def apply(self, x_list:list[int]):
        return int(np.dot([x_list[i] for i in self.indices], self.w_list) + self.b > 0)

class ProjectionPerceptron(Perceptron):
    def __init__(self, input_length, index, perceptron):
        super().__init__()
        self.input_length = input_length
        self.index = index
        self.perceptron = perceptron
    
    def apply(self, x_list:list[int]):
        return self.perceptron.apply([x_list[self.index] for i in range(self.perceptron.input_length)])


class TwoLayersPerceptron(Perceptron):
    def __init__(self, p_list:list[Perceptron], p:Type['Perceptron'], indices=None):
        super().__init__()
        
        self.input_length = p_list[0].input_length
        self.p = p
        self.indices = indices if indices is not None else list(range(len(p_list)))
        self.p_list = [p_list[i] for i in self.indices]
        # for pi in p_list:
        #     if self.input_length != pi.input_length:
        #         raise ValueError(f'The number of arguments for the elements of p_list are not aligned.')

    def apply(self, x_list):
        s_list = [pi.apply(x_list) for pi in self.p_list]
        return self.p.apply(s_list)
    


NAND = SimplePerceptron([-0.5, -0.5], 0.7)

NOT = ProjectionPerceptron(1, 0, NAND)

AND1 = TwoLayersPerceptron([NAND], NOT)
AND2 = TwoLayersPerceptron([NAND], NAND, [0, 0])
OR1 = TwoLayersPerceptron([ProjectionPerceptron(2, 0, NOT), ProjectionPerceptron(2, 1, NOT)], NAND)
OR2 = TwoLayersPerceptron([ProjectionPerceptron(2, 0, NAND), ProjectionPerceptron(2, 1, NAND)], NAND)

XOR = TwoLayersPerceptron([NAND, OR2], AND1)
print(XOR.apply([0,0]))
