from typing import Type


class Perceptron():
    def __init__(self):
        pass

    def apply(self, x1, x2):
        return 0
    
    def show_truth_table(self):
        print()


class SimplePerceptron(Perceptron):
    def __init__(self, w1, w2, b):
        super().__init__()
        self.w1 = w1
        self.w2 = w2
        self.b = b

    def apply(self, x1, x2):
        return int(x1 * self.w1 + x2 * self.w2 + self.b > 0)
    
    def show_truth_table(self):
        show_row = lambda x1, x2: f' {x1}| {x2}| {self.apply(x1, x2)}'
        print('\n'.join(['x1|x2|s1|s2| y', show_row(0, 0), show_row(0, 1), show_row(1, 0), show_row(1, 1)]))


class TwoLayersPerceptron(Perceptron):
    def __init__(self, p1:Type['SimplePerceptron'], p2:Type['SimplePerceptron'], p:Type['SimplePerceptron']):
        super().__init__()
        self.p1 = p1
        self.p2 = p2
        self.p3 = p

    def apply(self, x1, x2):
        s1 = self.p1.apply(x1, x2)
        s2 = self.p2.apply(x1, x2)
        return self.p3.apply(s1, s2)
    
    def show_truth_table(self):
        show_row = lambda x1, x2: f' {x1}| {x2}| {self.p1.apply(x1, x2)}| {self.p2.apply(x1, x2)}| {self.apply(x1, x2)}'
        print('\n'.join(['x1|x2|s1|s2| y', show_row(0, 0), show_row(0, 1), show_row(1, 0), show_row(1, 1)]))


# NAND = SimplePerceptron(-0.5, -0.5, 0.7)
# OR = SimplePerceptron(0.5, 0.5, -0.2)
# AND = SimplePerceptron(0.5, 0.5, -0.7)
# XOR = TwoLayersPerceptron(NAND, OR, AND)

# print('AND')
# AND.show_truth_table()
# print()
# print('NAND, OR → AND')
# XOR.show_truth_table()
