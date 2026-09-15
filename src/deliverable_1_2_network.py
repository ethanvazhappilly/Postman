"""1.2 - Building a small multi layer perceptron (MLP)"""

import numpy as np
 
from src.deliverable_1_1_layers import Linear, ReLU
 
 
class FeedForwardNetwork:
 
    def __init__(self, layer_sizes, seed=None):
        
        # Checking for correct layer sizes ie dimension of each layer
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes needs at least an input and an output size.")
 
        self.layer_sizes = list(layer_sizes)
        
        # We use seed to make weights reproducible across test runs, it fixes numpy's global random seed
        if seed is not None:
            np.random.seed(seed)  
 
        self.layers = []
        n_linear = len(layer_sizes) - 1

        # We do not use ReLU for the final output scores, here we check which layer we are on
        for i in range(n_linear):
            self.layers.append(Linear(layer_sizes[i], layer_sizes[i + 1]))
           
            if i < n_linear - 1:
                self.layers.append(ReLU())
 
    
    # Running the input through each layer and getting the final output
    def forward(self, x):
        
        # Converting input to a numpy array with float datatype
        out = np.asarray(x, dtype=float)

        for layer in self.layers:
            out = layer.forward(out)
        return out
 
    def forward_with_activations(self, x):
        """Same as forward(), but also returns the intermediate output of
        every layer. Useful for inspecting/debugging what each layer does."""
        out = np.asarray(x, dtype=float)
        intermediates = []
        for layer in self.layers:
            out = layer.forward(out)
            intermediates.append((repr(layer), out.copy()))
        return out, intermediates
 
    def num_parameters(self):
        # Gives total count of learnable numbers ie all weights + biases 
        total = 0
        for layer in self.layers:
            if isinstance(layer, Linear):
                total += layer.W.size + layer.b.size
        return total
 
if __name__ == "__main__":

    layer_sizes = [4, 8, 3]
    net = FeedForwardNetwork(layer_sizes=layer_sizes, seed=0)

    
    example_input = np.array([
        [1.0, 2.0, -1.0, 0.5],
        [0.0, -2.5, 3.1, -0.4]
    ])

    # Our forward pass implementation
    output = net.forward(example_input)

    print("*" * 60)
    print("--- 1.2 Forward Pass Output ---")
    print("*" * 60)

    print(f"Input Shape : {example_input.shape}")
    print(f"Output Shape: {output.shape}")
    print("\nRaw Logits Output:")
    print(output)
    print(f"\nTotal Network Parameters: {net.num_parameters()}")