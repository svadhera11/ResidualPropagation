# Residual Propagation: Understanding GNN Learning Dynamics

This is the official implementation of Residual Propagation (RP) proposed in ["How Graph Neural Networks Learn: Lessons from Training Dynamics"](https://arxiv.org/pdf/2310.05105), which is accepted to ICML 2024.

## Project Overview

This project investigates how Graph Neural Networks (GNNs) learn from data by analyzing their training dynamics. We propose Residual Propagation (RP), a novel algorithm inspired by the training dynamics of GNNs, which achieves competitive performance while being simpler and more interpretable than complex GNN architectures.

Our implementation includes:

1. **Basic RP Algorithm**: A simple yet effective label propagation approach that uses only the graph structure
2. **Generalized RP Algorithm**: An extension that incorporates node features through different kernel functions
3. **RK4 Integration**: Advanced numerical integration using Runge-Kutta methods
4. **GraphSAGE Integration**: A version that integrates with GraphSAGE for inductive learning
5. **Analysis Tools**: Scripts for analyzing experimental results and visualizing performance

## Implementation Details

The repository is organized into several key directories:

### 1. `RP/` - Basic Residual Propagation

This folder contains the implementation of the basic RP algorithm, which uses only the graph structure:

- `main.py`: Core implementation of the standard RP algorithm (Algorithm 1 in the paper)
- `main1.py` and `main2.py`: Variants with different experimental settings
- `dataset.py`: Dataset loading and processing
- `eval.py`: Evaluation metrics and utilities
- `utils.py`: Helper functions for graph processing, logging, etc.
- Analysis scripts: `analyze-log.py`, `analyze-log1.py`, `analyze-log2.py` for parsing and summarizing results

### 2. `Generalized_RP/` - Kernel-based RP

This folder contains the code for the generalized RP algorithm that incorporates node features:

- `main.py`: Implementation of the kernel-based RP algorithm (Algorithm 2 in the paper)
- Kernel implementations:
  - Gaussian Kernel
  - Cosine Kernel
  - Sigmoid Kernel

### 3. GraphSAGE Integration

- `RP/sage.py`: Complete implementation of GraphSAGE architecture for node classification
- `RP/run-graphsage.sh`: Script for running GraphSAGE on various datasets

## The Algorithm

The core RP algorithm is remarkably simple yet effective:

```python
# Initialize residuals as [Y,0]
residuals = torch.zeros_like(data.onehot_y)
residuals[train] = data.onehot_y[train].clone()

for step in range(1, args.steps + 1):
    # Create masked residuals [R,0]
    masked_residuals = residuals.clone()
    masked_residuals[test] = 0

    # Update residuals [R,R'] = [R,R'] - η LP([R,0])
    residuals -= step_size * LabelPropagation(masked_residuals)

predictions = -residuals[test]
```

The generalized version additionally uses kernel functions to incorporate node features:

```python
# Kernel matrix calculation
if args.kernel == 'gaussian':
    K = GaussianKernel(data.x, args.gamma)
elif args.kernel == 'cosine':
    K = CosineKernel(data.x)
elif args.kernel == 'sigmoid':
    K = SigmoidKernel(data.x, args.gamma)

# Label propagation now incorporates the kernel matrix
residuals -= args.lr * algorithm(masked_residuals, K)
```

## RK4 Integration

For more accurate numerical integration, we implement Runge-Kutta methods (RK4) for propagating information through the graph. This method provides higher-order accuracy compared to simple Euler integration:

```python
def rk4_update(f, y, h):
    """
    Fourth-order Runge-Kutta integration
    
    Args:
        f: Function to integrate
        y: Current state
        h: Step size
    
    Returns:
        Updated state
    """
    k1 = f(y)
    k2 = f(y + h * k1 / 2)
    k3 = f(y + h * k2 / 2)
    k4 = f(y + h * k3)
    
    return y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
```

This integration method improves the stability and accuracy of the propagation process, especially for complex graph structures.

## GraphSAGE Implementation

Our GraphSAGE implementation provides inductive learning capabilities:

```python
class GraphSageNet(nn.Module):
    """2‑ or 3‑layer GraphSAGE with ReLU + dropout."""
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 num_layers: int = 2,
                 dropout: float = 0.5):
        super().__init__()

        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x
```

This implementation allows for effective learning on large graphs and generalizes to unseen nodes.

## Dataset Support

The implementation supports various graph datasets:

- Standard citation networks: Cora, Citeseer, PubMed
- OGB datasets: ogbn-arxiv, etc.
- Microsoft Academic datasets: CS, Physics
- Heterophilous graphs: Squirrel, Film, Questions, etc.

## Running the Code

### Installation

After cloning the repo, create a virtual environment and install the required packages:

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Basic RP Algorithm

```bash
cd RP/
chmod +x run.sh  # Make the script executable
./run.sh
```

### Running the Generalized RP Algorithm

```bash
cd Generalized_RP/
chmod +x run.sh
./run.sh
```

### Running GraphSAGE

```bash
cd RP/
chmod +x run-graphsage.sh
./run-graphsage.sh
```

### Custom Dataset and Hyperparameters

Modify the scripts in each folder to customize:
- Datasets
- Model hyperparameters
- Training settings

Downloaded datasets will be automatically saved in the `data/` folder.

## Result Analysis

The repository includes tools for analyzing experimental results:

```bash
cd RP/
python analyze-log.py  # Basic analysis
python analyze-log1.py  # Summary statistics
python analyze-log2.py  # Detailed performance metrics
```

## Performance Highlights

Our algorithms achieve competitive performance across various benchmarks:

- **Basic RP**: Achieves accuracy comparable to complex GNNs on citation networks
- **Generalized RP**: Shows strong performance on heterophilous graphs with different kernel functions
- **RK4 Integration**: Provides more stable convergence and improved accuracy
- **GraphSAGE Integration**: Enables inductive learning with strong generalization

## Citation

If you find this code useful for your research, please cite our paper:

```
@author{,
  author={SagarVerma, ShivayVadhera, Mohiboddin, PanavShah},
}
```


## Contributing

Contributions to this repository are welcome. Please feel free to submit a Pull Request.



Downloaded datasets will be saved in the `data/` folder.

Change the scripts `run.sh` to modify the datasets or hyperparameters used.

