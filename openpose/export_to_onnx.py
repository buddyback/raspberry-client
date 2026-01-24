#!/usr/bin/env python3
"""
Export Lightweight OpenPose PyTorch model to ONNX format.

Run this script on your x86 development machine (where PyTorch works),
then copy the generated .onnx file to the Raspberry Pi.

Usage:
    python export_to_onnx.py

Output:
    openpose/checkpoint/openpose_lightweight.onnx
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.load_state import load_state


class OpenPoseWrapper(nn.Module):
    """
    Wrapper that extracts only the final stage heatmaps and PAFs.
    
    The original model outputs all intermediate stages, but we only need
    the last refinement stage outputs for inference.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        stages_output = self.model(x)
        # Return only the final stage outputs: heatmaps and PAFs
        # stages_output[-2] = final heatmaps (19 channels)
        # stages_output[-1] = final PAFs (38 channels)
        return stages_output[-2], stages_output[-1]


def export_to_onnx(
    checkpoint_path: str = "checkpoint/checkpoint_iter_370000.pth",
    output_path: str = "checkpoint/openpose_lightweight.onnx",
    input_height: int = 256,
    input_width: int = 456,  # Common aspect ratio for video
):
    """
    Export the Lightweight OpenPose model to ONNX format.
    
    Args:
        checkpoint_path: Path to the PyTorch checkpoint
        output_path: Path for the output ONNX file
        input_height: Input height for the model
        input_width: Input width for the model
    """
    print(f"Loading PyTorch model from {checkpoint_path}...")
    
    # Create model
    net = PoseEstimationWithMobileNet()
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    load_state(net, checkpoint)
    
    # Set to evaluation mode
    net = net.eval()
    
    # Wrap the model to get only final outputs
    wrapped_model = OpenPoseWrapper(net)
    wrapped_model = wrapped_model.eval()
    
    print(f"Model loaded successfully")
    
    # Create dummy input
    # Input shape: (batch, channels, height, width)
    dummy_input = torch.randn(1, 3, input_height, input_width)
    
    # Test the model first
    print("Testing PyTorch model...")
    with torch.no_grad():
        heatmaps, pafs = wrapped_model(dummy_input)
        print(f"  PyTorch heatmaps shape: {heatmaps.shape}")  # Should be [1, 19, H/8, W/8]
        print(f"  PyTorch PAFs shape: {pafs.shape}")  # Should be [1, 38, H/8, W/8]
    
    print(f"\nExporting to ONNX with input shape: {dummy_input.shape}...")
    
    # Use legacy export to avoid dynamo issues
    # Disable dynamo export by setting the environment variable
    os.environ["TORCH_ONNX_USE_EXPERIMENTAL_LOGIC"] = "0"
    
    # Export to ONNX using the legacy API
    torch.onnx.export(
        wrapped_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=12,  # Use opset 12 which is well-supported
        do_constant_folding=True,
        input_names=['input'],
        output_names=['heatmaps', 'pafs'],
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'heatmaps': {0: 'batch_size', 2: 'height', 3: 'width'},
            'pafs': {0: 'batch_size', 2: 'height', 3: 'width'},
        },
        dynamo=False,  # Explicitly disable dynamo
    )
    
    print(f"ONNX model saved to: {output_path}")
    
    # Verify the ONNX model
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verification passed!")
        
        # Print output info
        print("\nONNX model outputs:")
        for output in onnx_model.graph.output:
            print(f"  {output.name}: {[d.dim_value for d in output.type.tensor_type.shape.dim]}")
            
    except ImportError:
        print("(onnx package not installed, skipping verification)")
    except Exception as e:
        print(f"Warning: ONNX verification issue: {e}")
    
    # Test with ONNX Runtime
    try:
        import onnxruntime as ort
        import numpy as np
        
        print("\nTesting with ONNX Runtime...")
        session = ort.InferenceSession(output_path, providers=['CPUExecutionProvider'])
        
        # Check input/output names
        print(f"  Input name: {session.get_inputs()[0].name}")
        print(f"  Output names: {[o.name for o in session.get_outputs()]}")
        
        # Run inference
        test_input = np.random.randn(1, 3, input_height, input_width).astype(np.float32)
        outputs = session.run(None, {'input': test_input})
        
        print(f"  ONNX heatmaps output shape: {outputs[0].shape}")
        print(f"  ONNX PAFs output shape: {outputs[1].shape}")
        
        # Verify shapes match PyTorch
        assert outputs[0].shape == tuple(heatmaps.shape), "Heatmaps shape mismatch!"
        assert outputs[1].shape == tuple(pafs.shape), "PAFs shape mismatch!"
        
        print("ONNX Runtime test passed! Shapes match PyTorch output.")
        
    except ImportError:
        print("(onnxruntime not installed, skipping runtime test)")
    except Exception as e:
        print(f"Warning: ONNX Runtime test failed: {e}")
    
    print(f"\n✅ Done! Copy '{output_path}' to your Raspberry Pi.")
    return output_path


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    export_to_onnx()
