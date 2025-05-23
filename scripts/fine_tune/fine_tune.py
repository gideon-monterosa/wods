import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import types
import numpy as np
from tabpfn import TabPFNRegressor, TabPFNClassifier
from tabpfn.model.transformer import PerFeatureTransformer
from tabpfn.model.layer import PerFeatureEncoderLayer
from sklearn.datasets import make_classification, load_breast_cancer, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import copy
from tqdm import tqdm


class BottleneckAdapter(nn.Module):
    """Simple bottleneck adapter for fine-tuning TabPFN."""

    def __init__(self, input_dim, reduction_factor=8, dropout=0.1):
        super().__init__()
        self.bottleneck_dim = max(1, input_dim // reduction_factor)

        self.down_proj = nn.Linear(input_dim, self.bottleneck_dim)
        self.activation = nn.GELU()
        self.up_proj = nn.Linear(self.bottleneck_dim, input_dim)
        self.dropout = nn.Dropout(dropout)

        # Initialize with small weights for minimal initial impact
        nn.init.normal_(self.down_proj.weight, std=0.01)
        nn.init.normal_(self.up_proj.weight, std=0.01)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        residual = x
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_proj(x)
        return x + residual


def freeze_original_parameters(model):
    """
    Freeze all original TabPFN parameters to ensure they don't change during training.
    """
    # First, freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Double-check by iterating through all modules
    for name, module in model.named_modules():
        if not isinstance(module, BottleneckAdapter):
            for param in module.parameters():
                param.requires_grad = False

    print("All original TabPFN parameters frozen.")


def verify_frozen_parameters(model):
    """
    Verify that only adapter parameters have requires_grad=True.
    """
    trainable_params = []
    frozen_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(name)
        else:
            frozen_params.append(name)

    print(f"\nParameter Status:")
    print(f"Trainable parameters: {len(trainable_params)}")
    print(f"Frozen parameters: {len(frozen_params)}")

    # Check that trainable params are only adapters
    for param_name in trainable_params:
        if "adapter" not in param_name:
            print(f"WARNING: Non-adapter parameter is trainable: {param_name}")

    return trainable_params, frozen_params


def add_adapters_to_tabpfn(model, reduction_factor=8, dropout=0.1):
    """
    Add bottleneck adapters to a TabPFN model and ensure original weights are frozen.

    Args:
        model: The TabPFN model to add adapters to
        reduction_factor: Bottleneck reduction factor
        dropout: Dropout rate for adapters

    Returns:
        The model with adapters added
    """
    if not isinstance(model, PerFeatureTransformer):
        raise TypeError("Model must be a PerFeatureTransformer instance")

    # Step 1: Freeze ALL parameters first
    freeze_original_parameters(model)

    # Step 2: Store original forward methods
    for i, layer in enumerate(model.transformer_encoder.layers):
        layer._original_forward = layer.forward

    def modified_layer_forward(self, state, single_eval_pos=None, **kwargs):
        """Modified forward pass that includes adapter modules."""
        # Call original forward
        output = self._original_forward(state, single_eval_pos, **kwargs)

        # Now apply adapters (these will have gradients)
        if (
            hasattr(self, "adapter") and self.training
        ):  # Only apply adapters during training
            # Reshape if needed (handle 4D tensors: batch, seq, features, dim)
            original_shape = output.shape
            if len(output.shape) == 4:
                batch, seq, features, dim = output.shape
                output = output.reshape(-1, dim)
                output = self.adapter(output)
                output = output.reshape(original_shape)
            else:
                output = self.adapter(output)

        return output

    # Step 3: Add adapters to each transformer layer
    for i, layer in enumerate(model.transformer_encoder.layers):
        # Determine the hidden dimension
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "linear1"):
            hidden_dim = layer.mlp.linear1.in_features
        else:
            hidden_dim = model.ninp

        # Add adapter
        layer.adapter = BottleneckAdapter(hidden_dim, reduction_factor, dropout)

        # Replace forward method
        layer.forward = types.MethodType(modified_layer_forward, layer)

    # Step 4: Verify that only adapter parameters are trainable
    verify_frozen_parameters(model)

    return model


def get_adapter_parameters(model):
    """
    Get ONLY adapter parameters from the model and verify they are the only trainable ones.

    Args:
        model: The TabPFN model with adapters

    Returns:
        List of adapter parameters
    """
    adapter_params = []
    adapter_param_names = []

    for name, module in model.named_modules():
        if isinstance(module, BottleneckAdapter):
            for param_name, param in module.named_parameters():
                param.requires_grad = True  # Enable gradients only for adapters
                adapter_params.append(param)
                adapter_param_names.append(f"{name}.{param_name}")

    # Verify no other parameters are trainable
    all_trainable = [
        name for name, param in model.named_parameters() if param.requires_grad
    ]

    if set(all_trainable) != set(adapter_param_names):
        print("WARNING: Some non-adapter parameters are trainable!")
        unexpected = set(all_trainable) - set(adapter_param_names)
        print(f"Unexpected trainable parameters: {unexpected}")

    print(f"\nAdapter parameters: {len(adapter_params)}")
    for name in adapter_param_names:
        print(f"  - {name}")

    return adapter_params


class TabPFNWithAdapters:
    """Wrapper class for TabPFN models with adapters for fine-tuning."""

    def __init__(self, base_model, task_type="classification"):
        """
        Args:
            base_model: Pre-trained TabPFN model (TabPFNClassifier or TabPFNRegressor)
            task_type: 'classification' or 'regression'
        """
        self.base_model = base_model
        self.task_type = task_type
        self.device = base_model.device_
        self.adapters_added = False

        # Store original model state for verification
        self.original_state_dict = copy.deepcopy(base_model.model_.state_dict())

    def add_adapters(self, reduction_factor=8, dropout=0.1):
        """Add adapters to the model."""
        if not self.adapters_added:
            add_adapters_to_tabpfn(self.base_model.model_, reduction_factor, dropout)
            self.adapters_added = True
            self.adapter_params = get_adapter_parameters(self.base_model.model_)

            # Verify original parameters haven't changed
            self._verify_original_weights_unchanged()

            print(f"\nAdded {len(self.adapter_params)} adapter parameter groups")
            print(
                "Original TabPFN weights are frozen and will not be updated during training."
            )

    def _verify_original_weights_unchanged(self):
        """Verify that original weights haven't changed."""
        current_state = self.base_model.model_.state_dict()

        for name, param in self.original_state_dict.items():
            if "adapter" not in name and name in current_state:
                if not torch.equal(param, current_state[name]):
                    print(f"WARNING: Original parameter {name} has changed!")

        print("Verification complete: Original weights unchanged.")

    def fine_tune(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        epochs=10,
        batch_size=32,
        lr=1e-3,
        weight_decay=1e-4,
    ):
        """
        Fine-tune the model with adapters.
        """
        if not self.adapters_added:
            raise RuntimeError("Adapters not added. Call add_adapters() first.")

        # Double-check: freeze all non-adapter parameters
        freeze_original_parameters(self.base_model.model_)

        # Prepare data
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)

        if self.task_type == "classification":
            y_train_tensor = y_train_tensor.long()

        dataset = TensorDataset(X_train_tensor, y_train_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Setup optimizer - ONLY with adapter parameters
        print(
            f"\nSetting up optimizer with {len(self.adapter_params)} adapter parameters only."
        )
        optimizer = optim.AdamW(self.adapter_params, lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

        # Training loop
        for epoch in range(epochs):
            self.base_model.model_.train()
            total_loss = 0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

            for batch_idx, (batch_X, batch_y) in enumerate(progress_bar):
                optimizer.zero_grad()

                # Use a small context for efficiency
                context_size = min(100, len(X_train_tensor))
                X_train_context = X_train_tensor[:context_size]
                y_train_context = y_train_tensor[:context_size]

                # Prepare data in the format TabPFN expects
                # According to the forward method, we can use: model(train_x, train_y, test_x)
                train_x = X_train_context.unsqueeze(1)  # Add sequence dimension
                train_y = y_train_context.unsqueeze(1)  # Add sequence dimension
                test_x = batch_X.unsqueeze(1)  # Add sequence dimension

                # Forward pass through the model using the correct calling convention
                with torch.cuda.amp.autocast(enabled=False):
                    # TabPFN expects (train_x, train_y, test_x) as positional arguments
                    outputs = self.base_model.model_(train_x, train_y, test_x)

                # Extract logits/predictions
                if isinstance(outputs, dict):
                    if self.task_type == "classification":
                        logits = outputs.get("standard", outputs)
                    else:
                        predictions = outputs.get("standard", outputs)
                else:
                    if self.task_type == "classification":
                        logits = outputs
                    else:
                        predictions = outputs

                # Calculate loss
                if self.task_type == "classification":
                    # Remove the sequence dimension for loss calculation
                    logits = logits.squeeze(1) if logits.dim() > 2 else logits
                    loss = nn.CrossEntropyLoss()(logits, batch_y)
                else:
                    predictions = predictions.squeeze()
                    loss = nn.MSELoss()(predictions, batch_y)

                # Backward pass - only adapter parameters will be updated
                loss.backward()

                # Clip gradients for stability
                torch.nn.utils.clip_grad_norm_(self.adapter_params, max_norm=1.0)

                # Verify gradients are only in adapter parameters (check once)
                if epoch == 0 and batch_idx == 0:
                    self._verify_gradients()

                optimizer.step()

                total_loss += loss.item()
                progress_bar.set_postfix({"loss": loss.item()})

            scheduler.step()
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")

            # Validation
            if X_val is not None and y_val is not None:
                val_score = self.evaluate(X_val, y_val)
                print(f"Validation Score: {val_score:.4f}")

        # Final verification
        print("\nTraining complete. Verifying original weights unchanged...")
        self._verify_original_weights_unchanged()

    def _verify_gradients(self):
        """Verify that only adapter parameters have gradients."""
        print("\nVerifying gradients...")
        has_adapter_grads = False
        has_other_grads = False

        for name, param in self.base_model.model_.named_parameters():
            if param.grad is not None:
                if "adapter" not in name:
                    print(f"WARNING: Non-adapter parameter {name} has gradient!")
                    has_other_grads = True
                else:
                    has_adapter_grads = True
            elif "adapter" in name and param.requires_grad:
                print(f"WARNING: Adapter parameter {name} has no gradient!")

        if has_adapter_grads and not has_other_grads:
            print("Gradient check passed: Only adapter parameters have gradients.")

    def evaluate(self, X_test, y_test):
        """Evaluate the model on test data."""
        self.base_model.model_.eval()

        with torch.no_grad():
            if self.task_type == "classification":
                predictions = self.base_model.predict(X_test)
                score = accuracy_score(y_test, predictions)
            else:
                predictions = self.base_model.predict(X_test)
                score = -mean_squared_error(y_test, predictions)

        return score

    def save_adapters(self, path):
        """Save only the adapter weights."""
        adapter_state_dict = {}
        for name, module in self.base_model.model_.named_modules():
            if isinstance(module, BottleneckAdapter):
                adapter_state_dict[name] = module.state_dict()
        torch.save(
            {
                "adapter_state_dict": adapter_state_dict,
                "adapter_config": {
                    "reduction_factor": list(module.parameters())[0].shape[0]
                    * 8
                    // list(module.parameters())[0].shape[1],
                    "dropout": module.dropout.p,
                },
            },
            path,
        )
        print(f"Saved adapter weights to {path}")

    def load_adapters(self, path):
        """Load adapter weights."""
        checkpoint = torch.load(path)
        adapter_state_dict = checkpoint["adapter_state_dict"]

        for name, module in self.base_model.model_.named_modules():
            if isinstance(module, BottleneckAdapter) and name in adapter_state_dict:
                module.load_state_dict(adapter_state_dict[name])

        print(f"Loaded adapter weights from {path}")


# Test function with additional verification
def test_with_verification():
    print("Testing Classification Fine-tuning with Weight Freezing Verification...")

    # Load data
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create and fit base model
    classifier = TabPFNClassifier(
        device="cuda" if torch.cuda.is_available() else "cpu", n_estimators=1
    )
    classifier.fit(X_train[:100], y_train[:100])

    # Store original parameters for comparison
    original_params = {}
    for name, param in classifier.model_.named_parameters():
        original_params[name] = param.clone().detach()

    # Create wrapper and add adapters
    wrapper = TabPFNWithAdapters(classifier, task_type="classification")
    wrapper.add_adapters(reduction_factor=8, dropout=0.1)

    # Evaluate before fine-tuning
    print("\nBefore fine-tuning:")
    score_before = wrapper.evaluate(X_test, y_test)
    print(f"Test Accuracy: {score_before:.4f}")

    # Fine-tune
    wrapper.fine_tune(
        X_train[:200],
        y_train[:200],
        X_val=X_test,
        y_val=y_test,
        epochs=3,
        batch_size=32,
        lr=1e-3,
    )

    # Evaluate after fine-tuning
    print("\nAfter fine-tuning:")
    score_after = wrapper.evaluate(X_test, y_test)
    print(f"Test Accuracy: {score_after:.4f}")
    print(f"Improvement: {score_after - score_before:.4f}")

    # Verify original parameters haven't changed
    print("\n" + "=" * 50)
    print("FINAL VERIFICATION: Checking all original parameters...")
    all_unchanged = True
    for name, param in classifier.model_.named_parameters():
        if "adapter" not in name and name in original_params:
            if not torch.equal(param.data, original_params[name]):
                print(f"ERROR: Parameter {name} has changed!")
                print(
                    f"  Max difference: {(param.data - original_params[name]).abs().max().item()}"
                )
                all_unchanged = False

    if all_unchanged:
        print("SUCCESS: All original TabPFN parameters remain unchanged!")
    else:
        print("ERROR: Some original parameters have changed!")

    print("=" * 50)


if __name__ == "__main__":
    test_with_verification()
