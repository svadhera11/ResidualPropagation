# Problem Statement

The goal of this project is to replicate the results obtained in the paper, as well as perform a few ablation studies on them. While we are able to replicate the results as given in the paper, the ablations ask:

1. Can we express the RP iteration as a continuous-time ODE, and solve it using ODE solution methods? In particular we use RK4 with timestep of 0.2 to examine if it has any performance improvements.
2. Could we improve the performance of the model by incorporating a complement channel ($A^{c}$) containing global information? 

# Approach and Results

## Residual Propagation
The baseline RP equation is:
$$ \[ R_{t+1}, R^{`}_{t+1} \] = -\eta A^{k} \[ R_t, 0 \] + \[R_t, R^{`}_{t} \] $$

It is implemented in the following class in `main.py`:

        ```class LabelPropagation(nn.Module):
        '''
        based on paper 'Learning with Local and Global Consistency'
        input:
            redisuals: [instances, k], val and test instances are masked by 0
        output:
            y_pred: [instances, k]
        '''
        def __init__(self, k, alpha):
            super().__init__()
            self.k = k
            self.alpha = alpha
            self.A = None

        def preprocess(self, args, edge_index, num_nodes, num_classes):
            self.A = construct_sparse_adj(edge_index, num_nodes=num_nodes, type='DAD')

        def forward(self, redisuals):
            if self.alpha < 1: 
                y_pred = redisuals.clone()
                for _ in range(self.k):
                    y_pred = self.alpha * self.A @ y_pred + (1 - self.alpha) * redisuals
                return y_pred
            else:
                for _ in range(self.k):
                    redisuals = self.A @ redisuals
                return redisuals```

Here $A$ is the adjacency matrix of the graph we are interested in and $R_t$ is the residual of the training data (nodes) at time $t$. Test nodes have estimated residuals stored in $R^{'}_t$.

The ablation was run with a modified `run.sh` file. As suggested in the paper, we ran a hyperparameter search over learning rates (`lr`) from 0.01 to 1 in seven steps, and for each learning rate, we tried ten values of $k$ - from 1 to 10. Performance was exhaustively logged in `run_log.txt `, but only for steps where it improved more than 2 decimal places, since it was observed that for many steps the performance did not change much. In order to see how much better Generalized RP was than RP in heterophilic settings, we also ran it on datasets like `amazon-ratings ` and `roman-empire`. Our performance summaries are as below, with the quantities in brackets representing the values quoted in the paper. A * means that the quantities are quoted when running generalized RP on the dataset in the paper, but we have used vanilla RP for homophilic datasets here.

| Dataset        | Best Validation Accuracy | Best Test Accuracy       | Best Hyperparameters   |
|----------------|--------------------------|---------------------------|--------------------------|
| ogbn-arxiv     | 73.18 (71.37)            | 70.08 (70.06)            | lr = 0.05, k = 7         |
| cora           | 70.60                    | 71.40 (82.7*)            | lr = 0.02, k = 10        |
| citeseer       | 48.40                    | 48.00 (73.0*)            | lr = 0.01, k = 9         |
| pubmed         | 72.80                    | 71.90 (80.1.0*)            | lr = 0.02, k = 8         |
| roman-empire   | 8.44                     | 8.58 (66.01 ± 0.56*)     | lr = 1.0, k = 3          |
| amazon-ratings | 49.19                    | 48.64 (47.95 ± 0.57)     | lr = 0.05, k = 4         |


We see that vanilla RP can be competitive in some scenarios to Generalized RP, especially in homophilic graphs, though there are some situations where it is utterly abysmal (see roman-empire).

### Ablation A: Continuous-time RK4

We write the ODE as


        ```class RK4Propagation(nn.Module):
            def __init__(self, k, eta, dt=0.2):
                super().__init__()
                self.k = k
                self.eta = eta
                self.dt = dt
                self.A = None
                self.train_mask = None

            def preprocess(self, args, edge_index, num_nodes, train_mask):
                self.A = construct_sparse_adj(edge_index, num_nodes=num_nodes, type='DAD').to(args.device)
                self.train_mask = train_mask.to(args.device)

            def apply_Ak_B(self, r):
                # Apply B (mask out non-train nodes)
                r = r * self.train_mask.unsqueeze(-1)
                # Apply A k times
                for _ in range(self.k):
                    r = torch.sparse.mm(self.A, r)
                return -self.eta * r

            def propagate(self, r0):
                steps = int(1.0 / self.dt)
                out = [r0]
                r = r0
                for _ in range(steps):
                    k1 = self.apply_Ak_B(r)
                    k2 = self.apply_Ak_B(r + 0.5 * self.dt * k1)
                    k3 = self.apply_Ak_B(r + 0.5 * self.dt * k2)
                    k4 = self.apply_Ak_B(r + self.dt * k3)
                    r = r + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
                    out.append(r)
                return out```

Essentially replacing the discrete time update by $\frac{dx}{dt} = -eta BA^k x$ where $x$ is the vector of residuals, $B$ is the filter matrix that is a diagonal matrix with 1s for the training nodes and 0s everywhere else. In principle this is solvable exactly, but to do so ended up requiring massive amounts of memory (100 GB+ of RAM to store the dense matrices generated by $\exp(-\etaBAt)$). As a result we ended up using RK4. Nonethless, the idea was to split each discrete timestep into five timesteps of 0.2 units each, to see if increaing the resolution of the model would help its performance. 
In a nutshell, it did not help or harm performance, though it took twice as long to run. 

Here is a summary of its performance on different datasets. Here baseline refers to the results above.

| Dataset        | Best Val Acc (Baseline) | Best Val Acc (RK4) | Best Test Acc (Baseline) | Best Test Acc (RK4) | Best Hyperparams (Baseline) | Best Hyperparams (RK4) |
|----------------|-------------------------|---------------------|---------------------------|----------------------|------------------------------|--------------------------|
| ogbn-arxiv     | 73.18                   | 71.37               | 70.08                     | 70.07                | lr = 0.05, k = 7             | lr = 0.5, k = 7          |
| cora           | 70.60                   | 70.60               | 71.40                     | 71.40                | lr = 0.02, k = 10            | lr = 0.2, k = 10         |
| citeseer       | 48.40                   | 48.40               | 48.00                     | 48.00                | lr = 0.01, k = 9             | lr = 0.05, k = 9         |
| pubmed         | 72.80                   | 72.80               | 71.90                     | 71.90                | lr = 0.02, k = 8             | lr = 0.2, k = 8          |


### Ablation B: Complement Mixing

The idea of "complement mixing" is shown below: 

        ```class LabelPropagation(nn.Module):
            '''
            Label Propagation with:
            y ← α·A^k·y + (1−α)·A_comp^k·y
            where A_comp·y = sum(y)·1 − y − A·y
            '''
            def __init__(self, k, alpha):
                super().__init__()
                self.k = k
                self.alpha = alpha
                self.A = None

            def preprocess(self, args, edge_index, num_nodes, num_classes):
                self.A = construct_sparse_adj(edge_index, num_nodes=num_nodes, type='DAD').to(args.device)

            def apply_A_power(self, A, y, k):
                for _ in range(k):
                    y = torch.sparse.mm(A, y)
                return y

            def apply_A_comp_power(self, y, k):
                for _ in range(k):
                    sum_y = y.sum(dim=0, keepdim=True)
                    Ay = torch.sparse.mm(self.A, y)
                    y = sum_y - y - Ay
                return y

            def forward(self, redisuals):
                # One pass of combined propagation
                A_y      = self.apply_A_power(self.A, redisuals, self.k)
                Acomp_y  = self.apply_A_comp_power(redisuals, self.k)
                y_pred   = self.alpha * A_y + (1 - self.alpha) * Acomp_y
                return y_pred```

We swept alpha in the range 0.1 - 0.9 in increments of 0.1, and kept the same sweep over learning rate and k.

The performance for this ablation was abysmal, to say the least: 

| Dataset     | Best Validation Accuracy | Best Test Accuracy | Best Hyperparameters |
|-------------|---------------------------|--------------------|-----------------------|
| ogbn-arxiv  | 22.98                     | 21.56              | lr = 10.0, k = 1      |
| cora        | 45.20                     | 44.80              | lr = 0.05, k = 1      |
| citeseer    | 26.00                     | 29.40              | lr = 0.1, k = 1       |
| pubmed      | 42.60                     | 41.20              | lr = 0.1, k = 1       |

While the idea was to explore if simple injection of long-range and complementary signals could benefit Residual Propagation, it is clear that such an approach causes the model to perform poorly. It could simply be that the complement information is not used well in such a simple model, injecting little more than noise into the system. Moreover it is possible that the loss of local structure induced by such an update could harm more than nonlocal information helps.
In conclusion, while complement mixing is a natural approach to take further, it appears that RP is far too simple a model to implement it on. A new model that incorportes nonlocal signals from the start may be a better candidate altogether to see if simple RP-inspired models can achieve competitive performance with lower computational overheads.

## Generalized Residual Propagation

Following a structured hyperparameter sweep, we varied the learning rate (`lr`) over discrete ranges adapted to each dataset, based on the `run.sh` in the original files. For cora and citeseer, we explored `lr` values of 5.0, 10.0, and 15.0, with iteration counts `k` in {2, 3, 4}. For pubmed, a higher range of `lr` in {50, 75, 100, 125} was tested alongside `k` in {3, 4, 5}. For heterophilic graphs like roman-empire and amazon-ratings, lower learning rates (0.5 to 3.0) and shallower propagation depths (`k` in {0, 1}) were considered, and 10 runs were used to average performance. The kernel function was also varied between Gaussian, Cosine, and Sigmoid similarity matrices. All experiments were logged to `run_log.txt` (~80 MB), but only steps with more than 0.01 improvement in accuracy were stored, since changes across many iterations were marginal. Below, we report the best performance found for each dataset and kernel.


| Dataset            | Kernel   | Best Val Acc (%) | Best Test Acc (%) | Best Hyperparameters           | Paper Test Accuracy (%) |
| ------------------ | -------- | ---------------- | ----------------- | ------------------------------ | ----------------------- |
| cora          | cosine   | 80.00            | 81.20             | lr = 5.0, k = 4, gamma = 1.0   | 82.70 \*                |
|                    | gaussian | 79.80            | 82.00             | lr = 15.0, k = 3, gamma = 1.0  |                         |
|                    | sigmoid  | 79.20            | 80.30             | lr = 10.0, k = 4, gamma = 1.0  |                         |
| citeseer       | cosine   | 5.80             | 7.70              | lr = 5.0, k = 2, gamma = 20.0  | 73.00 \*                |
|                    | gaussian | 75.00            | 72.70             | lr = 10.0, k = 3, gamma = 20.0 |                         |
|                    | sigmoid  | 74.60            | 72.20             | lr = 15.0, k = 4, gamma = 20.0 |                         |
| pubmed         | cosine   | 83.40            | 80.20             | lr = 50.0, k = 5, gamma = 1.0  | 80.10 \*                |
|                    | gaussian | 82.60            | 79.90             | lr = 50.0, k = 5, gamma = 1.0  |                         |
|                    | sigmoid  | 82.00            | 79.60             | lr = 125.0, k = 4, gamma = 1.0 |                         |
| roman-empire   | cosine   | 4.17             | 4.17              | lr = 0.5, k = 0, gamma = 1.0   | 66.01 ± 0.56 \*         |
|                    | gaussian | 66.65            | 66.61             | lr = 1.0, k = 0, gamma = 1.0   |                         |
|                    | sigmoid  | 40.88            | 39.66             | lr = 1.0, k = 0, gamma = 1.0   |                         |
| amazon-ratings | cosine   | 42.30            | 41.12             | lr = 2.0, k = 1, gamma = 1.0   | 47.95 ± 0.57            |
|                    | gaussian | 48.78            | 48.26             | lr = 2.0, k = 0, gamma = 1.0   |                         |
|                    | sigmoid  | 40.52            | 39.47             | lr = 1.0, k = 1, gamma = 1.0   |                         |

Running the whole ablation took about 2 days and 6 hours on a RTX 3060 with 12 GB VRAM.

 
