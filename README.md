# scPII
<!-- [![Downloads](https://pepy.tech/badge/scPII)](https://pepy.tech/project/scPII) -->
[![Python](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FXenon8778%2FscPII%2Frefs%2Fheads%2Fmain%2Fpyproject.toml
)](https://img.shields.io/python/required-version-toml)
[![GitHub license](https://img.shields.io/github/license/Xenon8778/scPII.svg)](https://github.com/Xenon8778/scPII/LICENSE)


A Python package for perturbation response scanning on gene regulatory networks.



## Introduction
*Single-cell pertubation impact index (scPII)*, computes an Impact metric to quantify global effects of gene pertubations using perturbation response scanning (PRS) on gene regulatory networks.

<br>
<img src="./docs/assets/workflow.png" style="max-width:700px;width:100%" >
<br> 

## Installation
You can install this package directly from GitHub using pip:
```shell
pip install git+https://github.com/xenon8778/scPII.git
```
or install it manually from source:
```shell
git clone https://github.com/xenon8778/scPII.git
cd scPII
pip install .
```
## Usage
Check out the example notebook - [Here!](docs/notebooks/small_simulated_example.ipynb)
Please install Seaborn to generate figures used in example.
```python
pip install seaborn
```

Assume 'G' is a graph and 'A' is its adjacency matrix stored as pandas DataFrame representing your GRN. Here's a quick example of how to use the main function:

```python
import pandas as pd
import networkx as nx
from scPII.core import scPRS

G = nx.powerlaw_cluster_graph(25, 1, 0.6, seed=0)
inputnet = pd.DataFrame(nx.adjacency_matrix(G).todense())

# Run perturbation response scanning (PRS)
PRSout = scPRS(inputnet)

# Summary statistics for each gene in GRN is stored in Summary layer.
print(PRSout['Summary'].head())
```

## Differential Impact Analysis
```python
import pandas as pd
import networkx as nx
from scPII.core import scPRS, differentialPRS

# Generate 1st Graph - Control
G1 = nx.powerlaw_cluster_graph(100, 1, 0.6, seed=0)
inputnet1 = pd.DataFrame(nx.adjacency_matrix(G1).todense())
genes1 = inputnet1.index

# Generate 2nd Graph - Case
G2 = nx.powerlaw_cluster_graph(100, 1, 0.6, seed=5)
inputnet2 = pd.DataFrame(nx.adjacency_matrix(G2).todense())
genes2 = inputnet2.index

# Perform PRS on both graphs
PRSoutControl = scPRS(inputnet1, explainedV=0.5, gene_names = genes, getPval = True)
PRSoutCase = scPRS(inputnet2, explainedV=0.5, gene_names = genes, getPval = True) 

# Perform differential impact analysis
DI_result = differentialPRS(PRS_case = PRSoutCase,
                            PRS_control = PRSoutControl)
DI_result
```

<!-- ## Contributing
We welcome contributions! Please feel free to open an issue or submit a pull request. -->

## Benchmarking Results 
Benchmarking results against various GRN construction algorithms, explained variance cutoffs, and virtual KO tools are available at [https://doi.org/10.5281/zenodo.17713781](https://doi.org/10.5281/zenodo.17713781).



