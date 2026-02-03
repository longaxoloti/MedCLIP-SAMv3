"""
Demo: Using Frequency-Adapter with BiomedCLIP
==============================================

This script demonstrates how to load a pre-trained BiomedCLIP model,
inject frequency adapters, and set up for training.

Example workflow:
1. Load pre-trained BiomedCLIP
2. Inject adapters into last 3 encoder layers
3. Freeze backbone, unfreeze adapters
4. Set up optimizer with only adapter parameters
5. Train with DHN-NCE loss
"""

import torch
import torch.nn as nn
import sys
import os

# Add frequency_adapter to path
module_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, module_dir)

try:
    from biomedclip_adapter import (
        inject_frequency_adapters,
        get_adapter_parameters,
        freeze_backbone_unfreeze_adapter,
        verify_adapter_injection
    )
except ImportError as e:
    print(f"Error importing from biomedclip_adapter: {e}")
    print("Trying alternative imports...")
    from __init__ import (
        inject_frequency_adapters,
        get_adapter_parameters,
        freeze_backbone_unfreeze_adapter,
        verify_adapter_injection
    )


def demo_basic_usage():
    """Basic usage example."""
    print("\n" + "="*70)
    print("DEMO 1: Basic Adapter Injection")
    print("="*70)
    
    try:
        from transformers import AutoModel
        
        print("\n[1] Loading pre-trained BiomedCLIP...")
        try:
            model = AutoModel.from_pretrained(
                "chuhac/BiomedCLIP-vit-bert-hf",
                trust_remote_code=True
            )
            print("✓ Model loaded successfully")
        except (NameError, RuntimeError) as e:
            print(f"⚠ Could not load BiomedCLIP model: {e}")
            print("  This is a HuggingFace/transformers compatibility issue, not an adapter issue")
            print("  Skipping Demo 1")
            return None, None
        
        print("\n[2] Injecting frequency adapters into last 3 layers...")
        model = inject_frequency_adapters(
            model,
            adapter_layers=[9, 10, 11],
            rank_k=64,
            gamma_init=0.1,
            verbose=True
        )
        
        print("\n[3] Freezing backbone, unfreezing adapters...")
        freeze_backbone_unfreeze_adapter(model, verbose=True)
        
        print("\n[4] Setting up optimizer...")
        adapter_params = get_adapter_parameters(model)
        optimizer = torch.optim.AdamW(adapter_params, lr=1e-4, weight_decay=1e-4)
        print(f"✓ Optimizer created with {len(adapter_params)} trainable parameters")
        
        print("\n[5] Verifying injection...")
        verification = verify_adapter_injection(model)
        print(f"✓ Adapters found: {verification['adapters_found']}")
        print(f"✓ Gammas found: {verification['gammas_found']}")
        print(f"✓ Issues: {len(verification['issues'])}")
        
        return model, optimizer
        
    except ImportError as e:
        print(f"⚠ Could not load transformers library: {e}")
        print(f"  Skipping this demo")
        return None, None


def demo_mock_model():
    """Demo with mock model (for testing without HuggingFace)."""
    print("\n" + "="*70)
    print("DEMO 2: Adapter Injection on Mock Model")
    print("="*70)
    
    # Create a minimal mock structure that mimics BiomedCLIP
    class MockBiomedCLIPEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_dim = 768
            self.layer_norm = nn.LayerNorm(768)
            self.linear = nn.Linear(768, 768)
        
        def forward(self, x, attention_mask=None, output_attentions=False):
            x = self.layer_norm(x)
            x = self.linear(x)
            return (x,)
    
    class MockBiomedCLIPEncoderLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_dim = 768
            self.encoder = MockBiomedCLIPEncoder()
        
        def forward(self, hidden_states, attention_mask=None, output_attentions=False):
            return self.encoder(hidden_states, attention_mask, output_attentions)
    
    class MockBiomedCLIP(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision_model = nn.Module()
            self.vision_model.encoder = nn.Module()
            self.vision_model.encoder.layers = nn.ModuleList([
                MockBiomedCLIPEncoderLayer() for _ in range(12)
            ])
    
    print("\n[1] Creating mock BiomedCLIP model...")
    model = MockBiomedCLIP()
    print("✓ Mock model created")
    
    print("\n[2] Injecting adapters into layers [10, 11]...")
    model = inject_frequency_adapters(
        model,
        adapter_layers=[10, 11],
        rank_k=32,
        verbose=True
    )
    
    print("\n[3] Moving model to GPU if available...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"✓ Model on device: {device}")
    
    print("\n[4] Testing forward pass...")
    freeze_backbone_unfreeze_adapter(model, verbose=False)
    
    # Create dummy input
    x = torch.randn(2, 197, 768).to(device)  # Batch=2, Tokens=197 (14x14 patches + CLS), Dim=768
    
    # Forward through wrapped layer
    wrapped_layer = model.vision_model.encoder.layers[11]
    output = wrapped_layer(x)
    
    print(f"✓ Forward pass successful")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output[0].shape}")
    
    print("\n[5] Testing backward pass...")
    loss = output[0].sum()
    loss.backward()
    
    # Check gradients
    for name, param in wrapped_layer.named_parameters():
        if 'adapter' in name or 'gamma' in name:
            if param.grad is not None:
                print(f"✓ Gradient flow in {name}: {param.grad.abs().mean():.6f}")
    
    return model


def demo_training_loop():
    """Demonstrate a mini training loop."""
    print("\n" + "="*70)
    print("DEMO 3: Mini Training Loop")
    print("="*70)
    
    # Create mock model
    class SimpleMockLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_dim = 768
            self.linear = nn.Linear(768, 768)
        
        def forward(self, hidden_states, attention_mask=None, output_attentions=False):
            return (self.linear(hidden_states),)
    
    class SimpleMockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision_model = nn.Module()
            self.vision_model.encoder = nn.Module()
            self.vision_model.encoder.layers = nn.ModuleList([
                SimpleMockLayer() for _ in range(12)
            ])
    
    print("\n[1] Creating model and injecting adapters...")
    model = SimpleMockModel()
    model = inject_frequency_adapters(model, adapter_layers=[11], verbose=False)
    freeze_backbone_unfreeze_adapter(model, verbose=False)
    
    print("[2] Setting up training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    adapter_params = get_adapter_parameters(model)
    optimizer = torch.optim.AdamW(adapter_params, lr=1e-4)
    
    print(f"✓ Model on {device}")
    print(f"✓ Trainable params: {sum(p.numel() for p in adapter_params):,}")
    
    print("\n[3] Running 3 training steps...")
    for step in range(3):
        # Create dummy batch
        x = torch.randn(4, 197, 768, device=device)
        
        # Forward pass
        layer = model.vision_model.encoder.layers[11]
        output = layer(x)
        loss = output[0].sum() * 0.01  # Dummy loss
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"  Step {step+1}: Loss = {loss.item():.6f}")
    
    print("\n✓ Training loop completed successfully")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("Frequency-Adapter Demo Suite")
    print("="*70)
    
    # Demo 1: Real BiomedCLIP (if available)
    model_real, optimizer_real = demo_basic_usage()
    
    # Demo 2: Mock model
    model_mock = demo_mock_model()
    
    # Demo 3: Mini training loop
    demo_training_loop()
    
    print("\n" + "="*70)
    print("All demos completed!")
    print("="*70 + "\n")
