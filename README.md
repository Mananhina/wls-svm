# LS-SVM and WLS-SVM
This repository contains scikit-learn interface implementations of machine learning algorithms for classification and regression tasks, with support for CPU and GPU computing.

## Installation

### How to use notebooks

From current folder, do following:

```bash
# Install environment with `ipynb` group
uv sync --frozen --group ipynb

# Create ipykernel to run notebooks
uv run -m ipykernel install --user --name wls_svm --display-name "Python (wls-svm)"

# Run jupyter notebook
uv run jupyter notebook
```

## Theory
For a more detailed explanation of the methods, see docs/theory.md

## Tests
Examples of calling functions and working methods can be found in the notebooks/test.ipynb
## License

MIT
