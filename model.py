import numpy as np
from config import INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE

class NeuralNetwork:
    def __init__(self):
        self.W1 = np.random.randn(INPUT_SIZE, HIDDEN_SIZE) * 0.1
        self.W2 = np.random.randn(HIDDEN_SIZE, OUTPUT_SIZE) * 0.1

    def forward(self, x):
        x = np.tanh(np.dot(x, self.W1))
        x = np.dot(x, self.W2)
        return np.argmax(x)

    def clone(self):
        clone = NeuralNetwork()
        clone.W1 = self.W1.copy()
        clone.W2 = self.W2.copy()
        return clone

    def flatten(self):
        return np.concatenate([self.W1.flatten(), self.W2.flatten()])

    def load_from_flat(self, flat):
        w1_size = self.W1.size
        self.W1 = flat[:w1_size].reshape(self.W1.shape)
        self.W2 = flat[w1_size:].reshape(self.W2.shape)