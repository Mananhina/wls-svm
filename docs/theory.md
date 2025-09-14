
### Theoretical base of LS-SVM and WLS-SVM methods

#### ● Classic SVM
 The problem of binary classification is formulated as follows: it is necessary to separate objects $$\overrightarrow{x}_i \in X \subset \mathbb{R}^n$$ , each of which corresponds to one of two classes: $$y_i \in$$ Y = {−1; +1}. Let the number of pairs of the form $$(\overrightarrow{x}_i, y_i)$$ equals $$m$$.
**Support Vector Machine (SVM)** in classification is a set of algorithms that describe ways to build a classifier. Its main idea is to build a hyperplane that separates elements of different classes in the most optimal way. The standard equation of the hyperplane is: $$\langle \overrightarrow{\alpha}, \overrightarrow{x} \rangle +\beta = 0$$. The separation will occur according to the rule:
$$\langle \overrightarrow{\alpha}, \overrightarrow{x}_i \rangle +\beta \geq 1 \Rightarrow y_i=1,$$
$$\langle \overrightarrow{\alpha}, \overrightarrow{x}_i \rangle +\beta \leq -1 \Rightarrow y_i=-1$$,
what can be combined: $$y_i(\langle \overrightarrow{\alpha}, \overrightarrow{x}_i \rangle +\beta) \geq1$$.

To separate classes in the most optimal way means to construct a separating hyperplane so that the nearest objects of the classes are located as far from it as possible. It is known that to maximize the width of the separating strip, it is necessary to minimize the norm of its weights in $$L_2$$, which is equivalent to minimizing $$\langle \overrightarrow{\alpha}, \overrightarrow{\alpha} \rangle$$ in $$L_2$$.

Considering that most often the sample is not linearly separable, we introduce variables responsible for the error on the $$i$$-th object: $$\xi_i >0$$. The total error also needs to be minimized (with some regulating coefficient $$C > 0$$). Then we get an optimization problem of the following type:
$$\begin{cases}
\langle \overrightarrow{\alpha}, \overrightarrow{\alpha} \rangle +C\sum\limits_{i=1}^m\xi_i \rightarrow min, \\
y_i(\langle \overrightarrow{\alpha}, \overrightarrow{x}_i \rangle +\beta) \geq 1 - \xi_i, i=1,...,m, \\
\xi_i \geq 0, i=1,...,m.
\end{cases}$$
The resulting system is a quadratic programming problem. There are many different approaches to solving such problems, including both analytical and numerical ones, but we will use the classical approach using the Karush-Kuhn-Tucker theorem (KKT). Using it, we will obtain the important result that the weight vector $$\overrightarrow{\alpha}$$ is a linear combination of training vectors lying on the boundary of the dividing strip, and we will also move on to a new system:
$$\begin{cases}
\frac{1}{2} \sum\limits_{i=1}^m \sum\limits_{j=1}^m \lambda_i \lambda_j y_i y_j \langle \overrightarrow{x}_i, \overrightarrow{x}_j \rangle -\sum\limits_{i=1}^m\lambda_i \rightarrow min, \\
0 \leq \lambda_i \leq C, i=1,...,m\\
\sum\limits_{i=1}^m \lambda_i y_i = 0, i=1,...,m,
\end{cases}$$
where $$\lambda_i$$ are the Lagrange multipliers for the Lagrange function of the minimization problem. We replace the scalar product of vectors $$\langle \overrightarrow{x}_i, \overrightarrow{x}_j \rangle$$ with the result of the kernel function: $$K(\overrightarrow{x}_i, \overrightarrow{x}_j)$$.
$$\begin{cases}
\frac{1}{2} \sum\limits_{i=1}^m \sum\limits_{j=1}^m \lambda_i \lambda_j y_i y_j K(\overrightarrow{x}_i, \overrightarrow{x}_j) -\sum\limits_{i=1}^m\lambda_i \rightarrow min, \\
0 \leq \lambda_i \leq C, i=1,...,m\\
\sum\limits_{i=1}^m \lambda_i y_i = 0, i=1,...,m,
\end{cases}$$
The resulting problem is the minimization of a convex function on a convex set, i.e. it has a unique solution. The solution is the vector of Lagrange multipliers $$\overrightarrow{\lambda} = (\lambda_1, ..., \lambda_m)$$, from which the desired $$\overrightarrow{\alpha}$$ and $$\beta$$ are then constructed:
$$\overrightarrow{\alpha} = \sum\limits_{i=1}^m \lambda_i y_i \overrightarrow{x}_i$$, 
$$\beta=med(K(\overrightarrow{\alpha}, \overrightarrow{x}_j) - y_i: 0 < \lambda_i < C, i=1,...,m)$$.
The final classifier looks like this:
$$a(\overrightarrow{x}) = sign(\sum\limits_{i=1}^m \lambda_i y_i K(\overrightarrow{x}_i, \overrightarrow{x}_j) + \beta)$$.
**The disadvantage of this approach** is the complexity of solving the quadratic programming problem. There is no general analytical approach to the solution, only algorithms that take into account the features of the support vector method and, therefore, perform calculations more efficiently. But these methods still require time and computing power, which is especially noticeable with large amounts of data.

### ● LS-SVM for classification
**The Least Squares Support Vector Machine (LS-SVM)** is a modification of the classic support vector machine that introduces a new loss function to simplify the optimization problem.
Let's get rid of the inequality conditions $$\xi_i$$, which complicate the solution and cause a quadratic programming problem, and introduce only the equality conditions:
$$e_i = 1 - y_i(\langle \overrightarrow{\alpha}, \overrightarrow{x}_i \rangle +\beta), i=1,...,m$$.
Similarly, the sum of errors should be minimal, taking this into account, a new problem is formulated:
$$\begin{cases}
\langle \overrightarrow{\alpha}, \overrightarrow{\alpha} \rangle +C\sum\limits_{i=1}^me^2_i \rightarrow min, \\
e_i = 1 - y_i(\langle \overrightarrow{\alpha}, \overrightarrow{x}_i \rangle +\beta) , i=1,...,m. \\
\end{cases}$$
This system is an extremal problem with equality constraints. To solve it, we apply the Lagrange multiplier method.
The Lagrange function for the system: $$L = \lambda_0(\langle \overrightarrow{\alpha}, \overrightarrow{\alpha} \rangle +C\sum\limits_{i=1}^me^2_i) + \sum\limits_{i=1}^m \lambda_i(y_i(\langle \overrightarrow{\alpha}, \overrightarrow{x}_i \rangle +\beta) - 1 + e_i)$$, we take $$\lambda_0 = \frac{1}{2}$$, from the stationarity conditions we obtain:
$$\frac{\partial L}{\partial \overrightarrow{\alpha}} = \overrightarrow{\alpha} - \sum\limits_{i=1}^m \lambda_i y_i \overrightarrow{x}_i = 0 \Rightarrow \overrightarrow{\alpha} = \sum\limits_{i=1}^m \lambda_i y_i \overrightarrow{x}_i$$;
$$\frac{\partial L}{\partial \beta} = \sum\limits_{i=1}^m \lambda_i y_i = 0 \Rightarrow \overrightarrow{\alpha} = \sum\limits_{i=1}^m \lambda_i y_i \overrightarrow{x}_i$$;
$$\frac{\partial L}{\partial e_i} = Ce_i - \lambda_i = 0 \Rightarrow e_i = \frac{\lambda_i}{C}, i=1,...,m$$.
The condition $$e_i = 1 - y_i(\langle \overrightarrow{\alpha}, \overrightarrow{x}_i \rangle +\beta), i=1,...,m,$$ can be rewritten: $$1 = y_i\beta +y_i\sum\limits_{j=1}^m \lambda_j y_j \langle \overrightarrow{x}_j, \overrightarrow{x}_i \rangle +\frac{\lambda_i}{C}, i=1,...,m$$.
Then these equalities and the equality $$\sum\limits_{i=1}^m \lambda_i y_i = 0$$ can be written in matrix form:
$$\begin{pmatrix}
0 & \overrightarrow{y}^T \\
\overrightarrow{y} & R + \frac{1}{C}E
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
where $$R_{ij} = y_iy_j\langle \overrightarrow{x}_j, \overrightarrow{x}_i \rangle, i,j = 1,...,m,$$
$$\overrightarrow{1} = (1,...,1)^T, \overrightarrow{\lambda} = (\lambda_1,...,\lambda_m)^T, \overrightarrow{y} = (y_1,...,y_m)^T$$,
$$E = diag(1,...,1)$$.
Thus, the solution of the problem is reduced to solving a matrix equation. Its solution will be the vector of values $$\begin{pmatrix}
\beta \\
\overrightarrow{\lambda}
\end{pmatrix}$$. Since in the minimization problem the objective function is convex and the equality constraints form a convex set, the obtained solution is the absolute minimum of the problem. We can also, similar to the SVM approach, replace $$\langle \overrightarrow{x}_i, \overrightarrow{x}_j \rangle$$ with $$K(\overrightarrow{x}_i, \overrightarrow{x}_j)$$. We get the system:
$$\begin{pmatrix}
0 & \overrightarrow{y}^T \\
\overrightarrow{y} & Z + \frac{1}{C}E
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
where $$Z{ij} = y_iy_jK(\overrightarrow{x}_i, \overrightarrow{x}_j), i,j = 1,...,m$$.

**Thus, training with the LS-SVM method involves only solving the matrix equation (given above), which is algorithmically much simpler than solving the quadratic programming problem that arises when training with the SVM approach.**

It is important to note that the results in both approaches (SVM and LS-SVM) will not be the same. This is obvious from the formulations of these problems: the system in the first variant includes inequality conditions, the system in the second case includes only equality conditions. The method for solving the optimization problem using least squares is not an absolute replacement for solving the quadratic programming problem, it is only an alternative approach to the solution with its pros and cons.



