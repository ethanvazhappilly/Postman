"""Checking 1.2 against PyTorch"""

import numpy as np
import torch
import torch.nn as nn

from src.deliverable_1_1_layers import Linear
from src.deliverable_1_2_network import FeedForwardNetwork


def test_forward_pass_correctness():
    seed = 0
    layer_sizes = [784, 128, 64, 32, 10]
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Initializing our network
    custom_net = FeedForwardNetwork(layer_sizes=layer_sizes, seed=seed)

    # We create a test PyTorch network
    torch_net = nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 10)
    )

    # 4. Copy weights
    custom_linears = [l for l in custom_net.layers if isinstance(l, Linear)]
    torch_linears = [m for m in torch_net if isinstance(m, nn.Linear)]

    for c_layer, t_layer in zip(custom_linears, torch_linears):
        t_layer.weight.data = torch.tensor(c_layer.W.T, dtype=torch.float32)
        t_layer.bias.data = torch.tensor(c_layer.b, dtype=torch.float32)

    # 5. Dummy input batch matching layer_sizes[0] (784)
    x_numpy = np.random.randn(5, layer_sizes[0]).astype(np.float32)

    # 6. Forward passes
    custom_out = custom_net.forward(x_numpy)
    
    torch_net.eval()
    with torch.no_grad():
        torch_out = torch_net(torch.tensor(x_numpy)).numpy()

    # 7. Verification
    match = np.allclose(custom_out, torch_out, atol=1e-6)
    max_diff = np.max(np.abs(custom_out - torch_out))

    print("*" * 60)
    print("TEST RESULTS")
    print("*" * 60)
    print(f"Max Absolute Difference : {max_diff:.8f}")
    
    if match:
        print("STATUS: PASSED (NumPy output matches PyTorch reference)")
    else:
        print("STATUS: FAILED (Outputs exceed tolerance limit)")
    print("=" * 60)

    assert match, f"Forward pass outputs do not match PyTorch! Max diff: {max_diff}"


if __name__ == "__main__":
    test_forward_pass_correctness()