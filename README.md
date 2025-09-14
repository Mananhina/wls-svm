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
