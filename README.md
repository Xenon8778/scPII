# scPII
A Python package for perturbation response scanning on gene regulatory networks.

## Introduction
This package provides functionality to perform perturbation response scanning (PRS) on gene regulatory networks. It includes functions for eigen decomposition, PRS matrix computation, and summarizing key metrics for each gene.

## Installation
You can install this package directly from GitHub using pip:
```bash
pip install git+[https://github.com/xenon8778/scPII.git](https://github.com/xenon8778/scPII.git)
```
## Usage
Here's a quick example of how to use the main function:
```python
import pandas as pd
from scPII.scPII import scPRS

# Example usage (you'll need to provide your own data)
# Assume 'X' is a pandas DataFrame representing your gene regulatory network
# and 'gene_names' is a list of gene names.
# output = scPRS(X=my_data, gene_names=my_genes, Corr_cutoff=0.5)
# print(output['Summary'].head())
```

<!-- ## Contributing
We welcome contributions! Please feel free to open an issue or submit a pull request. -->
