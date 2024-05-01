from typing import Type
import numpy as np
import matplotlib.pyplot as plt


class Perceptron():
    def __init__(self):
        self.input_length = 0
        self.indices_list = None
    
    def apply(self, x_list:list[int], return_binary=True):
        return 0
    
    def show_truth_table(self):
        print()


class SimplePerceptron(Perceptron):
    def __init__(self, w_list:list[int], b):
        super().__init__()
        self.w_list = w_list
        self.b = b
        self.input_length = len(self.w_list)

    def apply(self, x_list:list[int], return_binary=True):
        if return_binary:
            return int(np.dot(x_list, self.w_list) + self.b > 0)
        else:
            return np.dot(x_list, self.w_list) + self.b


class ProjectionPerceptron(Perceptron):
    def __init__(self, input_length, indices, perceptron:Type['Perceptron']):
        super().__init__()
        self.input_length = input_length
        self.indices = indices
        self.perceptron = perceptron
    
    def apply(self, x_list:list[int], return_binary=True):
        return self.perceptron.apply([x_list[i] for i in self.indices], return_binary)


class TwoLayersPerceptron(Perceptron):
    def __init__(self, p_list:list[Perceptron], p:Type['Perceptron']):
        super().__init__()
        
        self.input_length = p_list[0].input_length
        self.p = p
        self.p_list = p_list

    def apply(self, x_list, return_binary=True):
        s_list = [pi.apply(x_list, return_binary) for pi in self.p_list]
        return self.p.apply(s_list, return_binary)


class ANDPerceptron(Perceptron):
    def __init__(self, p_list:list[Perceptron]):
        super().__init__()
        self.p_list = p_list
    
    def apply(self, x_list: list[int], return_binary=True):
        return np.array([p.apply(x_list, True) for p in self.p_list]).prod()


class XORPerceptron(Perceptron):
    def __init__(self, p_list:list[Perceptron]):
        super().__init__()
        self.p_list = p_list
    
    def apply(self, x_list: list[int], return_binary=True):
        return int(np.array([p.apply(x_list, True) for p in self.p_list]).sum() % 2)


def draw_perceptron_classification(perceptron:Type['Perceptron'], title='', title_font_size=18, plot_binary=False):
    fine_grid_points = np.mgrid[-3:3.1:0.05, -3:3.1:0.05].reshape(2, -1).T
    fine_outputs = np.array([perceptron.apply(point) for point in fine_grid_points])

    fine_points_0 = fine_grid_points[fine_outputs == 0]
    fine_points_1 = fine_grid_points[fine_outputs == 1]

    plt.figure(figsize=(6, 6))
    plt.scatter(fine_points_0[:, 0], fine_points_0[:, 1], color='#8888FF', label='Output 0')
    plt.scatter(fine_points_1[:, 0], fine_points_1[:, 1], color='#FF8888', label='Output 1')
    if plot_binary:
        plt.scatter([0, 0, 1, 1], [0, 1, 0, 1], color='#000000', s=100)
    plt.xticks(np.arange(-2, 3), fontsize=18)
    plt.yticks(np.arange(-2, 3), fontsize=18)
    plt.grid(True)
    plt.legend()
    plt.title(title, fontsize=title_font_size)
    plt.show()
