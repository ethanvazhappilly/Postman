"""1.1 - A small feedforward neural network with matrix operations"""


import numpy as np
 
 
class Linear:
    
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
 
        # W is the matrix containing the weights 
        self.W = np.random.randn(in_features, out_features) * 0.1

        # b is the matrix which contains the biases
        self.b = np.zeros(out_features)
 
    def forward(self, x):
        x = np.asarray(x, dtype=float)

        # We check if the matrix x has the right dimensions and if the number of columns (neurons) matches with what the layer expects
        if x.ndim != 2:
            raise ValueError("Expected 2D Array")
        if x.shape[1] != self.in_features:
            raise ValueError(
                f"Linear({self.in_features}, {self.out_features}) received input with "
                f"{x.shape[1]} features; expected {self.in_features}."
            )

        # We multiply matrix x with the weights and add the required biases    
        return x @ self.W + self.b
 
 
class ReLU:
    # This is the activation function used
    def forward(self, x):
        x = np.asarray(x, dtype=float)
        return np.maximum(0.0, x)
 
    def __repr__(self):
        return "ReLU()"
 
 
# Checking if the program was called directly
if __name__ == "__main__":
    
    print("*" * 60)
    print("1.1 - Neural network with basic layers")
    print("*" * 60)
 
    np.random.seed(0)

    # Just used as an example
    x = np.array([
        [0.5, 0.9, 0.1],
        [0.2, 0.1, 0.8],
        [0.0, 0.0, 1.0],
    ])
    print(f"\nExample Input:\n{x}")
 
    linear = Linear(3, 4)
    print(f"\nBuilt layer: {linear}")
 
    z = linear.forward(x)
    print(f"\nOutput, shape {z.shape}:\n{z}")
 
    relu = ReLU()
    a_relu = relu.forward(z)
    print(f"\nSame values after {relu}:\n{a_relu}")
 
