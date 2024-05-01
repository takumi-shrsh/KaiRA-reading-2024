class SimplePerceptron():
    def __init__(self, w1, w2, b):
        self.w1 = w1
        self.w2 = w2
        self.b = b

    def apply(self, x1, x2):
        return int(x1 * self.w1 + x2 * self.w2 + self.b > 0)
    
    def show_truth_table(self):
        print('\n'.join([
            'x1|x2| y',
            f' 0| 0| {self.apply(0, 0)}',
            f' 0| 1| {self.apply(0, 1)}',
            f' 1| 0| {self.apply(1, 0)}',
            f' 1| 1| {self.apply(1, 1)}',
            ]))


AND = SimplePerceptron(0.5, 0.5, -0.7)
OR = SimplePerceptron(0.5, 0.5, 0.2)
NAND = SimplePerceptron(-0.5, -0.5, 0.7)

# print(AND.apply(1, 1))
# NAND.show_truth_table()
