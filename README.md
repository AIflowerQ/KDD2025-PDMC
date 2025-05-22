# KDD2025 PDMC

This repository contains the code for the KDD 2025 paper *"PDMC: Generating Feasible Algorithmic Recourse via Perturbation Data Manifold Constraint"*. The project is still under active development, and we will continue to improve and update the repository.

## Example Data

All data-related materials are located in the `PDMC_data_process` folder. The notebook `example_data_generate.ipynb` demonstrates how to preprocess the data required for experiments, while the `example_data` directory provides a sample dataset that can be used directly.

## Runnable Example

The code for PDMC is located in the `PDMC_code` folder. The `ML_model_weights` directory contains the weights of the machine learning models targeted by the recourse. To train new models on a different dataset, please refer to the contents of the `predict_models` directory. The core implementation of PDMC can be found under `recourse_methods/catalog/OURS`, and a runnable example is provided in `runnable_example.py`.

## Contact

Feel free to reach out via email at wzm21@mails.tsinghua.edu.cn for any questions or discussions.