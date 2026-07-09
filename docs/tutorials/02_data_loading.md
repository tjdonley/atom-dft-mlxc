# Loading Data

This tutorial shows how to load and work with pre-computed atomic datasets.

## Overview

The `AtomicDataManager` class provides utilities for loading and managing atomic DFT datasets. This is useful for training machine learning models or analyzing pre-computed results.

## Basic Usage

```python
from atom.data import AtomicDataManager

# Initialize the data manager
manager = AtomicDataManager(
    data_root="./data_with_energy_density/",  # Path to your dataset
    scf_xc_functional="PBE0",
    forward_pass_xc_functionals=["GGA_PBE"],
)

print("Data manager initialized")
```

## Loading Data

Load data for a range of atomic numbers:

```python
# Load data for first 10 elements
dataset = manager.load_data(
    atomic_number_list=list(range(1, 11)),
    features_list=["rho", "grad_rho", "lap_rho"],
    use_radius_cutoff=False,
    print_summary=True,
)
```

## Exploring the Dataset

```python
# Check dataset properties
print(f"Number of configurations: {len(dataset.configuration_data_list)}")
print(f"Features available: {dataset.features_list}")
if dataset.atomic_numbers_per_sample is not None:
    print(f"Atomic numbers in dataset: {sorted(set(dataset.atomic_numbers_per_sample))}")
```

## Preparing Data Loaders

Prepare a data loader for training:

```python
# Prepare a potential data loader
potential_dataloader = dataset.prepare_potential_dataloader(
    target_functional="PBE0",
    target_component="v_xc",
    reference_functional=None,
    scale_features=True,
    scale_targets=True,
)

potential_dataloader.print_info()
```

## Note on Data Files

This tutorial assumes you have dataset files available. If you don't have data files:

1. You can generate data using `AtomicDataManager.generate_data()`
2. Or contact the authors for dataset access

For the public data APIs, see the [API Reference](../api/reference.md).
