import torch
import torch.nn as nn
import types
from tabpfn import TabPFNRegressor, TabPFNClassifier
from tabpfn.model.transformer import PerFeatureTransformer
from tabpfn.model.layer import PerFeatureEncoderLayer
from sklearn.datasets import make_classification, load_breast_cancer


class BottleneckAdapter(nn.Module):
    """Simple bottleneck adapter for fine-tuning TabPFN."""

    def __init__(self, input_dim, reduction_factor=8):
        super().__init__()

        self.bottleneck_dim = max(1, input_dim // reduction_factor)
        self.down_proj = nn.Linear(input_dim, self.bottleneck_dim)
        self.activation = nn.GELU()
        self.up_proj = nn.Linear(self.bottleneck_dim, input_dim)

        nn.init.normal_(self.down_proj.weight, std=0.01)
        nn.init.normal_(self.up_proj.weight, std=0.01)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        residual = x

        x = self.down_proj(x)
        x = self.activation(x)
        x = self.up_proj(x)

        return x + residual


def add_adapters_to_tabpfn(model, reduction_factor=8):
    """
    Add bottleneck adapters to a TabPFN model.

    Args:
        model: The TabPFN model to add adapters to
        reduction_factor: Bottleneck reduction factor

    Returns:
        The model with adapters added
    """
    if not isinstance(model, PerFeatureTransformer):
        raise TypeError("Model must be a PerFeatureTransformer instance")

    for param in model.parameters():
        param.requires_grad = False

    def modified_layer_forward(self, state, single_eval_pos=None, **kwargs):
        if not hasattr(self, "_original_forward"):
            self._original_forward = self.original_forward

        output = self._original_forward(state, single_eval_pos, **kwargs)

        if hasattr(self, "attention_adapter"):
            output = self.attention_adapter(output)

        if hasattr(self, "mlp_adapter"):
            output = self.mlp_adapter(output)

        return output

    for i, layer in enumerate(model.transformer_encoder.layers):
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "linear1"):
            input_dim = layer.mlp.linear1.in_features
        else:
            input_dim = model.ninp

        layer.attention_adapter = BottleneckAdapter(input_dim, reduction_factor)
        layer.mlp_adapter = BottleneckAdapter(input_dim, reduction_factor)

        # maybe save the original layer
        # layer.original_forward = layer.forward

        layer.forward = types.MethodType(modified_layer_forward, layer)

    return model


def get_adapter_parameters(model):
    """
    Get all adapter parameters from the model.

    Args:
        model: The TabPFN model with adapters

    Returns:
        List of adapter parameters
    """
    adapter_params = []
    for name, param in model.named_parameters():
        print(name)
        if "adapter" in name:
            param.requires_grad = True
            adapter_params.append(param)

    return adapter_params


X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)

classifier = TabPFNClassifier(device="cpu")
classifier.fit(X, y)
add_adapters_to_tabpfn(classifier.model_, reduction_factor=16)

classifier_adapter_params = get_adapter_parameters(classifier.model_)
print(classifier_adapter_params)

X, y = load_breast_cancer(return_X_y=True)
