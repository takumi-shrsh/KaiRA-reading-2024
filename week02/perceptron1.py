def perceptron(x1, x2, w1, w2, b):
    return int(x1 * w1 + x2 * w2 + b > 0)


def AND(x1, x2):
    return perceptron(x1, x2, 0.5, 0.5, -0.7)


def OR(x1, x2):
    return perceptron(x1, x2, 0.5, 0.5, 0.2)


def NAND(x1, x2):
    return perceptron(x1, x2, -0.5, -0.5, 0.7)


def XOR(x1, x2):
    return AND(NAND(x1, x2), OR(x1, x2))


# print(AND(1, 1))
