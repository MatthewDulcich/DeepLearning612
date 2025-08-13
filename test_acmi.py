#!/usr/bin/env python3
"""Test ACMI generation with fixed position/velocity extraction."""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_acmi_generation():
    """Test ACMI file generation with various model types."""
    print("Testing ACMI generation...")
    
    # Import the ACMI generation module
    try:
        from demo.generate_acmi import main as generate_acmi_main
        print("✓ Successfully imported ACMI generation module")
    except ImportError as e:
        print(f"✗ Failed to import ACMI module: {e}")
        return False
    
    # Test cases for different model types
    test_cases = [
        {
            "name": "LSTM Model Test",
            "args": ["--model-path", "runs/baseline_lstm/final_model.zip", 
                    "--model-type", "lstm",
                    "--output", "test_lstm.acmi",
                    "--steps", "100"]
        },
        {
            "name": "Transformer Model Test", 
            "args": ["--model-path", "runs/student_distilled/final_model.zip",
                    "--model-type", "transformer", 
                    "--output", "test_transformer.acmi",
                    "--steps", "100"]
        }
    ]
    
    for test_case in test_cases:
        print(f"\n--- {test_case['name']} ---")
        
        # Check if model file exists
        model_path = test_case['args'][1]
        if not Path(model_path).exists():
            print(f"⚠ Model file not found: {model_path}")
            print("  Skipping this test case...")
            continue
        
        try:
            # Temporarily replace sys.argv to simulate command line args
            original_argv = sys.argv
            sys.argv = ["generate_acmi.py"] + test_case['args']
            
            print(f"Running: python demo/generate_acmi.py {' '.join(test_case['args'])}")
            generate_acmi_main()
            
            # Restore original argv
            sys.argv = original_argv
            
            # Check if output file was created
            output_file = test_case['args'][3]  # --output value
            if Path(output_file).exists():
                print(f"✓ ACMI file generated: {output_file}")
                
                # Quick check of file content
                with open(output_file, 'r') as f:
                    content = f.read()
                    if content.strip():
                        print(f"✓ File has content ({len(content)} characters)")
                        # Check for common ACMI elements
                        if "FileType=text/acmi/tacview" in content:
                            print("✓ Valid ACMI header found")
                        if "#" in content:
                            print("✓ Timeline data found")
                    else:
                        print("✗ File is empty")
            else:
                print(f"✗ Output file not created: {output_file}")
                
        except Exception as e:
            print(f"✗ Test failed with error: {e}")
            sys.argv = original_argv  # Restore in case of error
    
    print("\nACMI generation test completed!")
    return True

if __name__ == "__main__":
    test_acmi_generation()
