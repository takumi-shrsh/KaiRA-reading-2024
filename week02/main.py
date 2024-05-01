import numpy as np
from perceptron import *


NAND = SimplePerceptron([-0.5, -0.5], 0.7)
NOT = ProjectionPerceptron(1, [0, 0], NAND)
AND = TwoLayersPerceptron([NAND], NOT)
OR = TwoLayersPerceptron([ProjectionPerceptron(2, [0], NOT), ProjectionPerceptron(2, [1], NOT)], NAND)
XOR = TwoLayersPerceptron([NAND, OR], AND)


def draw_star():
    def get_straight_line_parameter(x1, y1, x2, y2):
        if x1 == x2:
            return 1, 0, -x1
        m = (y2 - y1) / float(x2 - x1)
        intercept = -float(y1 - m * x1)
        return -m / intercept, 1 / intercept, 1

    five = lambda i: get_straight_line_parameter(
        2 * np.sin(2 * np.pi * i / 5),
        2 * np.cos(2 * np.pi * i / 5),
        2 * np.sin(2 * np.pi * (i + 2) / 5),
        2 * np.cos(2 * np.pi * (i + 2) / 5),
    )

    p_list = [SimplePerceptron(five(i)[:2], 1)  for i in range(0, 5)]
    perceptron = TwoLayersPerceptron([XORPerceptron(p_list)], NOT)
    draw_perceptron_classification(perceptron, 'Five straight lines and XOR gate')


def draw_parabola():
    def get_parameters_of_tangent_to_parabola(x):
        m = 2.0 * x
        y = x ** 2 - 2
        b = y - m * x
        return m / b, -1 / b, 1

    p_list = [SimplePerceptron(get_parameters_of_tangent_to_parabola(0.1 * i)[:2], (1 if i !=0 else 2))  for i in range(-20, 21)]
    perceptron = ANDPerceptron(p_list)
    draw_perceptron_classification(perceptron, 'Tangent lines of parabola and AND gate')


# draw_perceptron_classification(OR, 'XOR by NAND', plot_binary=True)
# draw_star()
# draw_parabola()
