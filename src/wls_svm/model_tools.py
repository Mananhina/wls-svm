"""Model tools."""
import codecs
import json
from typing import Literal

import numpy as np
import torch
from scipy.spatial.distance import cdist


def get_kernel(name, **params):
    """Get kernel function for input vectors with numpy computations.

    Args:
        name:
        **params:

    Returns:

    """

    def linear(x_i, x_j):
        return np.dot(x_i, x_j.T)

    def poly(x_i, x_j, d=params.get("d", 3)):
        return (np.dot(x_i, x_j.T) + 1) ** d

    def rbf(x_i, x_j, sigma=params.get("sigma", 1)):
        return np.exp(-(cdist(x_i, x_j) ** 2) / sigma**2)

    def sigmoid(x_i, x_j, alpha=params.get("alpha", 1), beta=params.get("beta", 1)):
        z = alpha * cdist(x_i, x_j) + beta
        return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

    kernels = {"linear": linear, "poly": poly, "rbf": rbf, "sigmoid": sigmoid}

    if kernels.get(name) is None:
        raise KeyError(
            f"Kernel '{name}' is not defined, try one in the list: "
            f"{list(kernels.keys())}."
        )
    else:
        return kernels[name]


def torch_get_kernel(
    name: Literal["linear", "poly", "rbf", "sigmoid"] = "sigmoid", **params
):
    """Get kernel function for input vectors with pytorch computations.

    Args:
        name:
        **params:

    Returns:

    """

    def linear(x_i, x_j):
        if isinstance(x_i, torch.Tensor) and isinstance(x_j, torch.Tensor):
            return torch.mm(x_i, torch.t(x_j))
        elif isinstance(x_i, np.ndarray) and isinstance(x_j, np.ndarray):
            x_i = torch.FloatTensor(x_i)
            x_j = torch.FloatTensor(x_j)
            return torch.mm(x_i, torch.t(x_j))
        else:
            raise ValueError("x_i, x_j in kernel must be torch.Tensor or np.ndarray")

    def poly(x_i, x_j, d=params.get("d", 3)):
        if isinstance(x_i, torch.Tensor) and isinstance(x_j, torch.Tensor):
            return (torch.mm(x_i, torch.t(x_j)) + 1) ** d
        elif isinstance(x_i, np.ndarray) and isinstance(x_j, np.ndarray):
            x_i = torch.FloatTensor(x_i)
            x_j = torch.FloatTensor(x_j)
            return (torch.mm(x_i, torch.t(x_j)) + 1) ** d
        else:
            raise ValueError("x_i, x_j in kernel must be torch.Tensor or np.ndarray")

    def rbf(x_i, x_j, sigma=params.get("sigma", 1)):
        if isinstance(x_i, torch.Tensor) and isinstance(x_j, torch.Tensor):
            return torch.exp(-(torch.cdist(x_i, x_j) ** 2) / sigma**2)
        elif isinstance(x_i, np.ndarray) and isinstance(x_j, np.ndarray):
            x_i = torch.FloatTensor(x_i)
            x_j = torch.FloatTensor(x_j)
            return torch.exp(-(torch.cdist(x_i, x_j) ** 2) / sigma**2)
        else:
            raise ValueError("x_i, x_j in kernel must be torch.Tensor or np.ndarray")

    def sigmoid(x_i, x_j, alpha=params.get("alpha", 1), beta=params.get("beta", 1)):
        if isinstance(x_i, torch.Tensor) and isinstance(x_j, torch.Tensor):
            z = alpha * cdist(x_i, x_j) + beta
            return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
        elif isinstance(x_i, np.ndarray) and isinstance(x_j, np.ndarray):
            x_i = torch.FloatTensor(x_i)
            x_j = torch.FloatTensor(x_j)
            z = alpha * cdist(x_i, x_j) + beta
            return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
        else:
            raise ValueError("x_i, x_j in kernel must be torch.Tensor or np.ndarray")

    kernels = {"linear": linear, "poly": poly, "rbf": rbf, "sigmoid": sigmoid}

    if kernels.get(name) is None:
        raise KeyError(
            f"Kernel '{name}' is not defined, try one in the list: "
            f"{list(kernels.keys())}."
        )
    else:
        return kernels[name]


def numpy_json_encoder(obj):
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    raise TypeError(f"""Unable to  "jsonify" object of type :', {type(obj)}""")


def torch_json_encoder(obj):
    if type(obj).__module__ == torch.__name__:
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        else:
            return obj.item()
    print(obj)
    raise TypeError(f"""Unable to  "jsonify" object of type :', {type(obj)}""")


def dump_model(model_dict, file_encoder, filepath="model"):
    with open(f"{filepath.replace('.json', '')}.json", "w") as fp:
        json.dump(model_dict, fp, default=file_encoder)


def load_model(filepath="model"):
    helper_filepath = filepath if filepath.endswith(".json") else f"{filepath}.json"
    file_text = codecs.open(helper_filepath, "r", encoding="utf-8").read()
    model_json = json.loads(file_text)

    return model_json
