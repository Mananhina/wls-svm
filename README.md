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
The main difference between LS-SVM and classical SVM is the approach to solving the optimization problem. Classic SVM solves the optimization problem:

$$\begin{cases}
\frac{1}{2} \sum\limits_{i=1}^m \sum\limits_{j=1}^m \lambda_i \lambda_j y_i y_j K(\overrightarrow{x}_i, \overrightarrow{x}_j) -\sum\limits_{i=1}^m\lambda_i \rightarrow min, \\
0 \leq \lambda_i \leq C, i=1,...,m,\\
\sum\limits_{i=1}^m \lambda_i y_i = 0, i=1,...,m,
\end{cases}$$

obtained by transitioning from a nonlinear programming problem.\
The main difficulty lies in finding the optimal Lagrange multipliers $$\lambda_i$$, which account for the impact of the error on the $$i$$-th object, since the errors themselves are defined by inequalities. Solving this optimization problem presents the greatest challenges.\
By specifying errors through equalities rather than non-equalities, we obtain an extremum problem with equality constraints. It can be reformulated in matrix form for classification:

$$\begin{pmatrix}
0 & \overrightarrow{y}^T \\
\overrightarrow{y} & y_iy_jK(\overrightarrow{x}_i, \overrightarrow{x}_j) + \frac{1}{C}I
\end{pmatrix}
\cdot
\begin{pmatrix}
\beta \\
\overrightarrow{\lambda}
\end{pmatrix} = 
\begin{pmatrix}
0 \\
\overrightarrow{1}
\end{pmatrix},$$

and for regression:

$$\begin{pmatrix}
0 & \overrightarrow{1}^T \\
\overrightarrow{1} & K(\overrightarrow{x}_i, \overrightarrow{x}_j) + \frac{1}{C}I
\end{pmatrix}
\cdot
\begin{pmatrix}
\beta \\
\overrightarrow{\lambda}
\end{pmatrix} = 
\begin{pmatrix}
0 \\
\overrightarrow{y}
\end{pmatrix}.$$

The solution to such problems can easily be obtained analytically.\
In WLS-SVM approach, instead of a matrix I, a matrix 
$$diag(v_1,...,v_n)$$, where $$v_i$$ is a weight correcting error on object $$i$$, is used.
\
\
For a more detailed explanation of the methods, see docs/theory.md

## Tests
Examples of calling functions and working methods can be found in the notebooks/test.ipynb
## License

MIT
