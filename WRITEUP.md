# Learnings from This Small Project

**Note:** I heard about the Postman recruitments last week and hence I couldn’t complete the whole thing in time. Most of my time was spent wrapping my head around how a neural network worked and learning about coding in python.

---

## 1. Initial Approach 

While first reading the task list it seemed to be very demanding. I had picked task 1 as it seemed the most interesting. I went through 3b1b’s video series on neural networks to gain a base understanding. 

I only knew basic python initially so I had spent time learning about object oriented programming and how classes and methods worked. 

I also learnt a bit about numpy and pytorch.

---

## 2. Understanding Neural Networks

Let us take the classic example of figuring out what number is drawn on an image.

It turns out that a low resolution image of a number can be easily recognised by the brain but it is difficult for a computer to recognise it correctly. We need to build a neural network to emulate the brain and feed it training data. There are various kinds of neural networks and this is one of the simplest ones.

Neural networks are made up of neurons which can be thought of as a container that holds a number between 0 and 1. The number is called the activation of the neuron.

* **First Layer**: The first layer of the network just consists of neurons which hold the grayscale value of each pixel of the image.
* **Intermediate Layers**: The activations in one layer will determine the activations in the next and so on.
* **Final Output Layer**: In the final output layer, the neuron with the highest activation is the network’s prediction of what the number is.

---

## 3. Activation Functions

For a particular neuron, we can tweak the weights and biases which can be thought of as knobs and dials for each connection between this neuron and the previous layer of neurons.

We take the weighted sum plus the bias and pass it to a function which squishes the output into our desired range of values..

* **Sigmoid Function**: One function to use would be the sigmoid function given by $1 / (1 + e^{-x})$ where more negative inputs tend to 0 and more positive inputs tend to 1. 
* **ReLU Function**: Another possible function we use is the ReLU which is easier to train. It stands for Rectified Linear Unit and its just a maximum of 0 or 'a' where a is the input. Hence, I ended up using this as the activation function. 

Thus the activation is just a measure of how positive the weighted sum is. The bias is just to ensure that the weighted sum is at a sufficiently high value before the neuron gets meaningfully active.

Each neuron in the layer has its own weight and its own biases. This means that there is a huge number of weights and biases when you count the whole network with all its layers.

---

## 4. Matrix Formulation

The above operations can be written as matrices:

* **Weight Matrix ($W$)**: Organized with dimensions `(in_features, out_features)`. Weights were initialized from a standard Gaussian distribution scaled down by $0.1$ (`np.random.randn(in_features, out_features) * 0.1`).
* **Bias Vector ($b$)**: Organized with dimensions `(1, out_features)`. Biases were initialized as a 2D row vector filled with zeros (`np.zeros((1, out_features))`).
* **Input Data ($X$)**: Structured as `(batch_size, in_features)`, where rows represent individual samples and columns represent feature values.

We compute the linear transformation by:

Z = X * W + b
