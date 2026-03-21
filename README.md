# MNIST Handwritten Digit Classifier (C Engine)

A high-performance Multi-Layer Perceptron (MLP) implemented in C. This project features a dynamically configurable architecture, an efficient IDX data loader, and a JSON export mechanism for deploying models to web environments.

## Key Features

- **Dynamic Topology**: Support for an arbitrary number of hidden layers and neurons. Configuration is handled via a simple integer array.
- **Advanced Math**:
  - **He Initialization**: Optimized weight starting values for ReLU activation.
  - **ReLU Activation**: Standard ReLU for hidden layers to prevent vanishing gradients.
  - **Softmax + Cross-Entropy**: Robust multi-class classification at the output layer.
- **Optimized Training**: Mini-batch Gradient Descent with pre-allocated delta buffers for high-speed backpropagation.
- **Portability**: Exports final weights and biases to `model_weights.json` for instant use in JavaScript/Web applications.
- **Performance**: Trained on the full 60,000 MNIST image set with high convergence stability.

---

## Technical Architecture

### 1. Flexible Network Core
Unlike standard "from-scratch" tutorials, this engine uses a dynamic pointer-to-pointer structure:
- `double **neurons`: Stores activations for every layer.
- `double **weights`: Stores the weight matrices connecting $L_n \to L_{n+1}$.
- `double **biases`: Stores the bias vectors for each transition.

### 2. The Training Pipeline
The `Trainer` struct manages the learning lifecycle:
- **Forward Pass**: Iterative matrix multiplication using a custom `matrix_multiply` utility.
- **Backpropagation**: Calculates deltas using the chain rule, moving from the output layer back to the input.
- **Gradient Accumulation**: Sums gradients across a batch before applying updates, reducing noise and increasing speed.

---

## Benchmarks

The model was verified on an **Intel i5-13420H** with the following configuration:
- **Topology**: [784, 128, 64, 10]
- **Dataset**: Full 60,000 MNIST training samples
- **Hyperparameters**: 
  - Learning Rate: 0.01
  - Batch Size: 64
  - Epochs: 15

| Metric | Result |
| :--- | :--- |
| **Training Time** | ~659 Seconds (11 Minutes) |
| **Final Loss** | **0.1663** |
| **Estimated Accuracy** | **~96.5%** |

---

## Build & Usage

### Prerequisites
- GCC Compiler
- Math library (`-lm`)

### Compilation
Navigate to the source directory and use the provided Makefile:
```bash
cd src
make clean && make
```

### Execution
Ensure the MNIST dataset is located in the `dataset/` directory at the project root.
```bash
./main.bin
```

---

## Project Structure

- `src/`: Core implementation files (`network.c`, `trainer.c`, `data_loader.c`, `utils.c`).
- `include/`: Header files defining the public API.
- `dataset/`: MNIST binary IDX files.
- `model_weights.json`: The exported model (generated after training).

