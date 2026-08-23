"""
Unit tests for the CIFAR-10 model and dataset modules.
"""

import sys
from pathlib import Path

import torch
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from model import get_model


class TestModel:
    """Tests for the model factory and architecture."""

    def test_resnet18_creation(self):
        """Test that ResNet-18 model can be created."""
        model = get_model(architecture="resnet18", num_classes=10)
        assert model is not None

    def test_resnet18_output_shape(self):
        """Test that model output shape matches num_classes."""
        model = get_model(architecture="resnet18", num_classes=10)
        model.eval()

        # CIFAR-10 input: batch=2, channels=3, height=32, width=32
        dummy_input = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            output = model(dummy_input)

        assert output.shape == (2, 10), f"Expected (2, 10), got {output.shape}"

    def test_resnet18_single_image(self):
        """Test forward pass with a single image."""
        model = get_model(architecture="resnet18", num_classes=10)
        model.eval()

        dummy_input = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            output = model(dummy_input)

        assert output.shape == (1, 10)

    def test_resnet18_cifar_adaptations(self):
        """Test that CIFAR-10 adaptations are applied correctly."""
        model = get_model(architecture="resnet18", num_classes=10)

        # Check conv1 uses 3x3 kernel (not 7x7)
        assert model.conv1.kernel_size == (3, 3)
        assert model.conv1.stride == (1, 1)
        assert model.conv1.padding == (1, 1)

        # Check maxpool is removed (replaced with Identity)
        assert isinstance(model.maxpool, torch.nn.Identity)

    def test_unsupported_architecture_raises(self):
        """Test that unsupported architecture raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported architecture"):
            get_model(architecture="vgg16", num_classes=10)

    def test_different_num_classes(self):
        """Test model with different number of output classes."""
        model = get_model(architecture="resnet18", num_classes=5)
        model.eval()

        dummy_input = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            output = model(dummy_input)

        assert output.shape == (1, 5)

    def test_model_is_trainable(self):
        """Test that the model can compute gradients."""
        model = get_model(architecture="resnet18", num_classes=10)
        model.train()

        dummy_input = torch.randn(2, 3, 32, 32)
        dummy_target = torch.randint(0, 10, (2,))

        output = model(dummy_input)
        loss = torch.nn.CrossEntropyLoss()(output, dummy_target)
        loss.backward()

        # Check that gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
                break
