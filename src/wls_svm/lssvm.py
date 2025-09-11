import numpy as np
import torch

from wls_svm.model_tools import (
    dump_model,
    get_kernel,
    load_model,
    numpy_json_encoder,
    torch_get_kernel,
    torch_json_encoder,
)


class LSSVM:
    """Parent class for WLS-SVC and WLS-SVR methods.

    # Parameters:
    - C: float, default = 1.0
        Regularization parameter that controls the trade-off between margin width and error minimization.
        Accepts values in the range (0, +infinity). Higher values of C result in stricter prediction.
    - kernel_name: {'linear', 'poly', 'rbf', 'sigmoid'}, default = 'rbf'
        Kernel function used to capture non-linear patterns.
    - on_GPU: boolean, default = False
        Flag determining where matrix operations will be performed:
        if True - on GPU, if False - on CPU.
    - kernel_params: **kwargs, default = depends on the selected kernel
        For 'linear' kernel no parameters;
        For 'poly' kernel parameter 'd' defines polynomial degree, default = 3;
        For 'rbf' kernel parameter 'sigma' - radius of Gaussian function, default = 1;
        For 'sigmoid' kernel parameters 'alpha' and 'beta' - parameters of sigmoid(x1, x2) = tanh(alpha*kernel(x1, x2) + beta),
        tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x)), default = 1 for both.

    # Attributes:
    - alpha: ndarray of shape (1, n_support_vectors) or (n_classes, n_support_vectors) for classification
        or (1, n_support_vectors) for regression.
        Weight vector (or set of vectors for multi-class classification)
    - b: ndarray of shape (1,) or (n_classes,) for classification
        or (1,) for regression.
        Bias term of the weight vector.
    - sv_x: ndarray of shape (n_support_vectors, n_features)
        Feature values of support vectors (data used to train the model).
    - sv_y: ndarray of shape (n_support_vectors, n_classes) for classification
        (n_support_vectors, 1) for regression.
        Target values of support vectors (data used to train the model).
    - K: function, default = rbf()
        Kernel function.
    """

    def __init__(self, C=1, kernel="rbf", on_GPU=False, **kernel_params):
        self.C = C
        self.on_GPU = on_GPU
        self.kernel_name = kernel
        self.kernel_params = kernel_params
        self.type = None

        # Training parameters
        self.alpha = None
        self.b = None
        self.sv_x = None
        self.sv_y = None

        if on_GPU:
            self.K = torch_get_kernel(kernel, **kernel_params)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.K = get_kernel(kernel, **kernel_params)
            self.device = None

    def _solve_SLE_with_cg(self, A, b, **params):
        """
        Helper function for solving a system of linear equations Ax = b
        using the conjugate gradient method.
        """

        max_iter = params.get("max_iter", 200)
        tol = params.get("tol", 1e-6)

        x = torch.zeros_like(b)
        r = b - A @ x
        p = r.clone()
        rsold = r @ r

        for _ in range(max_iter):
            Ap = A @ p
            alpha = rsold / (p @ Ap + 1e-10)
            x = x + alpha * p
            r = r - alpha * Ap
            rsnew = r @ r

            if torch.sqrt(rsnew) < tol:
                break

            p = r + (rsnew / rsold) * p
            rsold = rsnew

        return x

    def save(self, filepath="model", only_hyperparams=False):
        """Function for saving model parameters.
        - filepath: string, default = 'model'
            Path to json file for saving the model.
        - only_hyperparams: boolean, default = False
            Save only model parameters (False)
            or also training parameters (True).
        """

        model_json = {
            "type": self.type,
            "hyperparameters": {
                "C": self.C,
                "on_GPU": self.on_GPU,
                "kernel_name": self.kernel_name,
                "kernel_params": self.kernel_params,
            },
        }

        if (self.alpha is not None) and (not only_hyperparams):
            model_json["parameters"] = {
                "alpha": self.alpha,
                "b": self.b,
                "sv_x": self.sv_x,
                "sv_y": self.sv_y,
            }
            if hasattr(self, "y_labels"):
                model_json["parameters"]["y_labels"] = self.y_labels

        file_encoder = torch_json_encoder if self.on_GPU else numpy_json_encoder
        dump_model(model_dict=model_json, file_encoder=file_encoder, filepath=filepath)


class LSSVR(LSSVM):
    """WLS-SVR model class solving regression problems using Least Squares Support Vector Machine.

    Model configuration involves solving a system of linear equations of the form:
     _                                   _   _      _    _  _
    | 0              1_N^T               |  |   b   |   | 0 |
    |                                    |  |       | = |   |
    | 1_N   K(x_i, x_j) + dig(1/(C*v_i)) |  | alpha |   | y |
    |_                                  _|  |_     _|   |_ _|.

    Methods for solving SLE:
        when working on GPU: Gaussian elimination, conjugate gradient method;
        when working on CPU: Gaussian elimination.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.type = "LSSVR"

    def _construct_and_solve_SLE_torch(self, K, y, weights, solve_SLE_CG, **params):
        """Constructs the required SLE considering weights and solves it using the specified method;
        matrix operations are performed on GPU.
        """

        n = K.shape[0]
        Ones = torch.ones((n, 1), device=self.device)

        # Constructing SLE for finding model parameters
        C_inv = torch.diag(1 / (self.C * weights))

        A = torch.zeros((n + 1, n + 1), device=self.device)
        A[0, 1:] = Ones.T
        A[1:, 0] = Ones.squeeze()
        A[1:, 1:] = K + C_inv

        B = torch.cat([torch.zeros(1, device=self.device), y])

        if solve_SLE_CG:
            # Solve using conjugate gradient method
            solution = self._solve_SLE_with_cg(A, B, **params)
        else:
            # Solve using Gaussian elimination
            solution = torch.linalg.solve(A, B)

        return solution

    def _optimize_parameters_GPU(
        self,
        X,
        y_values,
        weighted,
        n_iter_weight,
        weight_param,
        solve_SLE_CG,
        **solve_sle_params,
    ):
        """
        Method for finding model parameters using GPU computations.
        """

        # Transfer target variable of training set to GPU
        y = torch.FloatTensor(y_values).to(self.device)

        # Matrix of kernel function values for all pairs of training objects, transfer to GPU
        K = torch.FloatTensor(self.K(X, X)).to(self.device)

        # Get first solution (LS-SVR)
        solution = self._construct_and_solve_SLE_torch(
            K,
            y,
            torch.ones(len(y_values), device=self.device),
            solve_SLE_CG,
            **solve_sle_params,
        )

        b, alpha = solution[0], solution[1:]

        if weighted:
            for _ in range(n_iter_weight):
                # Calculate errors and weights based on obtained solution
                preds = K @ alpha + b
                errors = y - preds
                weights = 1 / (1 + weight_param * errors**2)

                # Get subsequent solutions (WLS-SVR)
                solution = self._construct_and_solve_SLE_torch(
                    K, y, weights, solve_SLE_CG, **solve_sle_params
                )

                b, alpha = solution[0], solution[1:]

        return b, alpha

    def _construct_and_solve_SLE(self, K, y, weights):
        """Constructs the required SLE considering weights and solves it using Gaussian elimination;
        matrix operations remain on CPU.
        """

        Ones = np.array([[1]] * len(y))

        # Constructing SLE for finding model parameters
        A = np.block(
            [[0, Ones.T], [Ones, K + (self.C * weights) ** -1 * np.eye(len(y))]]
        )
        B = np.concatenate((np.array([0]), y))

        # Solve SLE using built-in Gaussian elimination
        solution = np.linalg.solve(A, B)

        return solution

    def _optimize_parameters(self, X, y_values, weighted, n_iter_weight, weight_param):
        """
        Method for finding model parameters using CPU computations.
        """

        K = self.K(
            X, X
        )  # Matrix of kernel function values for all pairs of training objects

        # Get first solution (LS-SVR)
        solution = self._construct_and_solve_SLE(K, y_values, np.ones(len(y_values)))

        b = solution[0]
        alpha = solution[1:]

        if weighted:
            for _ in range(n_iter_weight):
                # Calculate errors and weights based on obtained solution
                errors = np.zeros(y_values.shape[0])
                for i in range(y_values.shape[0]):
                    errors[i] = y_values[i] - np.sum(alpha * K[:, i]) - b

                weights = 1 / (1 + weight_param * errors**2)

                # Get subsequent solutions (WLS-SVR)
                solution_w = self._construct_and_solve_SLE(K, y_values, weights)

                b = solution_w[0]
                alpha = solution_w[1:]

        return b, alpha

    def fit(
        self,
        X,
        y,
        weighted=False,
        n_iter_weight=2,
        weight_param=1.0,
        solve_SLE_CG=False,
        **solve_sle_params,
    ):
        """Training the WLS-SVR model.
        - X: ndarray of shape (n_samples, n_features).
        - y: ndarray of shape (n_samples,).
        - weighted: boolean, default = False
            flag determining whether to apply error weighting procedure (WLS-SVR) if True
            or not to apply (LS-SVR) if False.
        - n_iter_weight: int, default = 2
            number of iterations for applying weighting procedure.
        - weight_param: float, default = 1.0
            regularization parameter in the weighting function,
            higher parameter value means higher penalty with increasing error value.
        - solve_SLE_CG: boolean, default = False
            flag determining the method for solving SLE (only for GPU version):
            if True - solve using conjugate gradient method
            if False - solve using Gaussian elimination.
        - **solve_sle_params: additional parameters for configuring conjugate gradient method, may include
            - max_iter: int, default = 200
                maximum number of iterations until convergence.
            - tol: float, default = 1e-6
                tolerance for method convergence.
        """

        # Save data used to train the model (all training data are support vectors)
        # and call optimization methods - methods for constructing and solving SLE
        if self.on_GPU:
            self.sv_x = torch.FloatTensor(X).to(self.device)
            self.sv_y = torch.FloatTensor(y).to(self.device)
            self.b, self.alpha = self._optimize_parameters_GPU(
                X,
                y,
                weighted,
                n_iter_weight,
                weight_param,
                solve_SLE_CG,
                **solve_sle_params,
            )
        else:
            self.sv_x = X
            self.sv_y = y
            self.b, self.alpha = self._optimize_parameters(
                X, y, weighted, n_iter_weight, weight_param
            )

    def predict(self, X):
        """Predicting target values using the trained model.
        - X: ndarray of shape (n_samples, n_features)
        """

        if self.alpha is None or self.b is None:
            raise Exception("Model is not fitted yet!")

        if self.on_GPU:
            X_reshaped = X.reshape(1, -1) if X.ndim == 1 else X
            X_gpu = torch.FloatTensor(X_reshaped).to(self.device)
            KxX = self.K(self.sv_x, X_gpu)

            pred = (self.alpha @ KxX + self.b).cpu().numpy()

        else:
            X_reshaped = X.reshape(1, -1) if X.ndim == 1 else X
            KxX = self.K(self.sv_x, X_reshaped)

            pred = np.dot(self.alpha, KxX) + self.b

        return pred

    @classmethod
    def load(rgs, filepath, only_hyperparams=False):
        """Method for loading model from .json file.
        - filepath: string
            Path to .json model file.
        - only_hyperparams: boolean, default = False
            Load only model parameters (False) or also training parameters (True).
        """

        model_json = load_model(filepath=filepath)

        if model_json["type"] == "LSSVR":
            lssvr = rgs(
                C=model_json["hyperparameters"]["C"],
                kernel=model_json["hyperparameters"]["kernel_name"],
                **model_json["hyperparameters"]["kernel_params"],
            )

        else:
            raise Exception(f"Model type '{model_json['type']}' doesn't match 'LSSVR'")

        if (model_json.get("parameters") is not None) and (not only_hyperparams):
            lssvr.alpha = np.array(model_json["parameters"]["alpha"])
            lssvr.b = np.array(model_json["parameters"]["b"])
            lssvr.sv_x = np.array(model_json["parameters"]["sv_x"])
            lssvr.sv_y = np.array(model_json["parameters"]["sv_y"])
            lssvr.y_values = np.array(model_json["parameters"]["y_values"])

        return lssvr


class LSSVC(LSSVM):
    """WLS-SVC model class solving classification problems using Least Squares Support Vector Machine.
    Approach for multi-class classification: One-vs-All

    Model configuration involves solving a system of linear equations of the form:
     _                                         _   _      _    _    _
    | 0                       y^T              |  |   b   |   |  0  |
    |                                          |  |       | = |     |
    | y    y_i*y_jK(x_i, x_j) + dig(1/(C*v_i)) |  | alpha |   | 1_N |
    |_                                        _|  |_     _|   |_   _|.

    Methods for solving SLE:
        when working on GPU: Gaussian elimination, conjugate gradient method;
        when working on CPU: Gaussian elimination.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.type = "LSSVC"
        self.y_labels = None

    def _construct_and_solve_SLE_torch(self, Omega, y, weights, solve_SLE_CG, **params):
        """Constructs the required SLE considering weights and solves it using the specified method;
        matrix operations are performed on GPU.
        """

        n = Omega.shape[0]

        # Constructing SLE for finding model parameters
        C_inv = torch.diag(1 / (self.C * weights))

        A = torch.zeros((n + 1, n + 1), device=self.device)
        A[0, 1:] = y.T
        A[1:, 0] = y.flatten()
        A[1:, 1:] = Omega + C_inv

        B = torch.zeros(n + 1, device=self.device)
        B[1:] = 1.0

        if solve_SLE_CG:
            # Solve using conjugate gradient method
            solution = self._solve_SLE_with_cg(A, B, **params)
        else:
            # Solve using Gaussian elimination
            solution = torch.linalg.solve(A, B)

        return solution

    def _optimize_parameters_GPU(
        self,
        X,
        y_values,
        weighted,
        n_iter_weight,
        weight_param,
        solve_SLE_CG,
        **solve_sle_params,
    ):
        """
        Method for finding model parameters using GPU computations.
        """

        # Transfer target variable of training set to GPU
        y = torch.FloatTensor(y_values).to(self.device)

        # Matrix of kernel function values for all pairs of training objects, transfer to GPU
        K = torch.FloatTensor(self.K(X, X)).to(self.device)

        Omega = y @ y.T * K  # Matrix used in SLE construction

        # Get first solution (LS-SVC)
        solution = self._construct_and_solve_SLE_torch(
            Omega,
            y,
            torch.ones(len(y_values), device=self.device),
            solve_SLE_CG,
            **solve_sle_params,
        )

        b, alpha = solution[0], solution[1:]

        if weighted:
            for _ in range(n_iter_weight):
                # Calculate errors and weights based on obtained solution
                preds = torch.sign(K @ (alpha * y.flatten()) - b)
                errors = y.flatten() - preds
                weights = 1 / (1 + weight_param * errors**2)

                # Get subsequent solutions (WLS-SVC)
                solution_w = self._construct_and_solve_SLE_torch(
                    Omega, y, weights, solve_SLE_CG, **solve_sle_params
                )

                b, alpha = solution_w[0], solution_w[1:]

        return b, alpha

    def _construct_and_solve_SLE(self, Omega, y, weights):
        """Constructs the required SLE considering weights and solves it using Gaussian elimination;
        matrix operations remain on CPU.
        """

        # Constructing SLE for finding model parameters
        A = np.block([[0, y.T], [y, Omega + (self.C * weights) ** -1 * np.eye(len(y))]])
        B = np.array([0] + [1] * len(y))

        # Solve SLE using built-in Gaussian elimination
        solution = np.linalg.solve(A, B)

        return solution

    def _optimize_parameters(self, X, y_values, weighted, n_iter_weight, weight_param):
        """
        Method for finding model parameters using CPU computations.
        """

        K = self.K(
            X, X
        )  # Matrix of kernel function values for all pairs of training objects
        Omega = np.multiply(y_values * y_values.T, K)  # Matrix used in SLE construction

        # Get first solution (LS-SVC)
        solution = self._construct_and_solve_SLE(
            Omega, y_values, np.ones(len(y_values))
        )

        b = solution[0]
        alpha = solution[1:]

        if weighted:
            for _ in range(n_iter_weight):
                # Calculate errors and weights based on obtained solution
                errors = np.zeros(y_values.shape[0])
                for i in range(y_values.shape[0]):
                    errors[i] = y_values[i] - np.sign(
                        np.sum(alpha * y_values * K[:, i]) - b
                    )

                weights = 1 / (1 + weight_param * errors**2)

                # Get subsequent solutions (WLS-SVC)
                solution_w = self._construct_and_solve_SLE(Omega, y_values, weights)

                b = solution_w[0]
                alpha = solution_w[1:]

        return b, alpha

    def fit(
        self,
        X,
        y,
        weighted=False,
        n_iter_weight=2,
        weight_param=1,
        solve_SLE_CG=False,
        **solve_sle_params,
    ):
        """Training the WLS-SVC model.
        - X: ndarray of shape (n_samples, n_features)
        - y: ndarray of shape (n_samples,).
        - weighted: boolean, default = False
            flag determining whether to apply error weighting procedure (WLS-SVR) if True
            or not to apply (LS-SVR) if False.
        - n_iter_weight: int, default = 2
            number of iterations for applying weighting procedure.
        - weight_param: float, default = 1.0
            regularization parameter in the weighting function,
            higher parameter value means higher penalty with increasing error value.
        - solve_SLE_CG: boolean, default = False
            flag determining the method for solving SLE (only for GPU version):
            if True - solve using conjugate gradient method
            if False - solve using Gaussian elimination.
        - **solve_sle_params: additional parameters for configuring conjugate gradient method, may include
            - max_iter: int, default = 200
                maximum number of iterations until convergence.
            - tol: float, default = 1e-6
                tolerance for method convergence.
        """

        y_reshaped = y.reshape(-1, 1) if y.ndim == 1 else y

        self.y_labels = np.unique(y_reshaped, axis=0)

        if len(self.y_labels) == 2:  # Binary classification
            # Convert labels to values {-1, +1}
            y_labels = np.where((y_reshaped == self.y_labels[0]).all(axis=1), -1, +1)[
                :, np.newaxis
            ]  # make it a column

            # Save data used to train the model (all training data are support vectors)
            # and call optimization methods - methods for constructing and solving SLE
            if self.on_GPU:
                self.sv_x = torch.FloatTensor(X).to(self.device)
                self.sv_y = torch.FloatTensor(y_reshaped).to(self.device)
                self.y_labels = torch.FloatTensor(self.y_labels).to(self.device)
                self.b, self.alpha = self._optimize_parameters_GPU(
                    X,
                    y_labels,
                    weighted,
                    n_iter_weight,
                    weight_param,
                    solve_SLE_CG,
                    **solve_sle_params,
                )
            else:
                self.sv_x = X
                self.sv_y = y_reshaped
                self.b, self.alpha = self._optimize_parameters(
                    X, y_labels, weighted, n_iter_weight, weight_param
                )

        else:  # Multi-class classification, One-vs-All approach
            n_classes = len(self.y_labels)
            self.b = np.zeros(n_classes)
            self.alpha = np.zeros((n_classes, len(y_reshaped)))

            for i in range(
                n_classes
            ):  # Binary classification for each individual class
                # convert labels: +1 for current class, -1 for other classes
                y_labels = np.where(
                    (y_reshaped == self.y_labels[i]).all(axis=1), +1, -1
                )[:, np.newaxis]

                # Save data used to train the model (all training data are support vectors)
                # and call optimization methods - methods for constructing and solving SLE
                if self.on_GPU:
                    self.b[i], self.alpha[i] = self._optimize_parameters_GPU(
                        X,
                        y_labels,
                        weighted,
                        n_iter_weight,
                        weight_param,
                        solve_SLE_CG,
                        **solve_sle_params,
                    )
                else:
                    self.b[i], self.alpha[i] = self._optimize_parameters(
                        X, y_labels, weighted, n_iter_weight, weight_param
                    )

    def predict(self, X):
        """Predicting class labels using the trained model.
        For binary classification, class labels are converted to values {-1, +1}.
        For multi-class classification, One-vs-All approach is used, for each individual class
        a binary classification problem is solved (also with class labels converted to values {-1, +1})
        - X: ndarray of shape (n_samples, n_features)
        """

        if self.alpha is None or self.b is None:
            raise Exception("Model is not fitted yet!")

        if self.on_GPU:
            X_reshaped = X.reshape(1, -1) if X.ndim == 1 else X

            X = torch.FloatTensor(X_reshaped).to(self.device)
            KxX = self.K(self.sv_x, X)

            if len(self.y_labels) == 2:  # Binary classification
                # Convert labels to values {-1, +1}
                y_values = torch.where(
                    (self.sv_y == self.y_labels[0]).all(axis=1),
                    torch.tensor(-1, dtype=X.dtype, device=self.device),
                    torch.tensor(+1, dtype=X.dtype, device=self.device),
                )

                # Prediction from the trained model
                y = torch.sign(
                    torch.mm((self.alpha * y_values).view(1, -1), KxX) + self.b
                )

                # Convert labels from {-1, +1} to original labels
                y_pred_labels = torch.where(
                    y == -1, self.y_labels[0], self.y_labels[1]
                ).view(-1)

            else:  # Multi-class classification, One-vs-All approach
                y = torch.empty(
                    (len(self.y_labels), len(X)), dtype=X.dtype, device=self.device
                )
                for i in range(len(self.y_labels)):
                    # convert labels: +1 for current class, -1 for other classes
                    y_values = torch.where(
                        (self.sv_y == self.y_labels[i]).all(axis=1),
                        torch.tensor(+1, dtype=X.dtype, device=self.device),
                        torch.tensor(-1, dtype=X.dtype, device=self.device),
                    )

                    # Prediction from all classifiers for the considered object
                    y[i] = (
                        torch.mm((self.alpha[i] * y_values).view(1, -1), KxX)
                        + self.b[i]
                    )

                # Final prediction - class with the most "confident answer",
                # i.e., with the maximum value of the classifier function
                # (distance to the separating hyperplane is maximum)
                predictions = torch.argmax(y, axis=0)
                y_pred_labels = torch.stack([self.y_labels[i] for i in predictions])

            return y_pred_labels.cpu().numpy()

        else:
            X = np.array(X)
            KxX = self.K(self.sv_x, X)

            if len(self.y_labels) == 2:  # Binary classification
                # Convert labels to values {-1, +1}
                y_values = np.where(
                    (self.sv_y == self.y_labels[0]).all(axis=1), -1, +1
                )[:, np.newaxis]

                # Prediction from the trained model
                y = np.sign(np.dot((self.alpha * y_values.flatten()), KxX) + self.b)

                # Convert labels from {-1, +1} to original labels
                y_pred = np.where(y == -1, self.y_labels[0], self.y_labels[1])

            else:  # Multi-class classification, One-vs-All approach
                y = np.zeros((len(self.y_labels), X.shape[0]))

                for i in range(len(self.y_labels)):
                    # convert labels: +1 for current class, -1 for other classes
                    y_values = np.where(
                        (self.sv_y == self.y_labels[i]).all(axis=1), +1, -1
                    )[:, np.newaxis]

                    # Prediction from all classifiers for the considered object
                    y[i] = np.dot((self.alpha[i] * y_values.flatten()), KxX) + self.b[i]

                # Final prediction - class with the most "confident answer",
                # i.e., with the maximum value of the classifier function
                # (distance to the separating hyperplane is maximum)
                predictions = np.argmax(y, axis=0)
                y_pred = np.array([self.y_labels[i] for i in predictions])

            return y_pred

    @classmethod
    def load(cls, filepath, only_hyperparams=False):
        """Method for loading model from .json file.
        - filepath: string
            Path to .json model file.
        - only_hyperparams: boolean, default = False
            Load only model parameters (False) or also training parameters (True).
        """

        model_json = load_model(filepath=filepath)

        if model_json["type"] == "LSSVC":
            lssvc = cls(
                C=model_json["hyperparameters"]["C"],
                kernel=model_json["hyperparameters"]["kernel_name"],
                **model_json["hyperparameters"]["kernel_params"],
            )

        else:
            raise Exception(f"Model type '{model_json['type']}' doesn't match 'LSSVC'")

        if (model_json.get("parameters") is not None) and (not only_hyperparams):
            lssvc.alpha = np.array(model_json["parameters"]["alpha"])
            lssvc.b = np.array(model_json["parameters"]["b"])
            lssvc.sv_x = np.array(model_json["parameters"]["sv_x"])
            lssvc.sv_y = np.array(model_json["parameters"]["sv_y"])
            lssvc.y_labels = np.array(model_json["parameters"]["y_labels"])

        return lssvc
