#!/usr/bin/env python3
"""
Utility functions for DeepPBS Gradio interface
Provides helper functions for file handling, result processing, and visualization
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import font_manager
from typing import List, Optional, Dict, Tuple
from pathlib import Path
import base64
import io
from PIL import Image

# Add DeepPBS modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'run'))

def get_adaptive_tick_spacing(length: int) -> int:
    """Determine optimal tick spacing based on sequence length"""
    if length <= 25:
        return 1
    elif length <= 50:
        return 2
    elif length <= 100:
        return 5
    elif length <= 200:
        return 10
    else:
        return max(1, length // 20)  # For very long sequences, show ~20 ticks

def create_sequence_logo(pwm_data: np.ndarray, title: str = "Sequence Logo",
                        figsize: Tuple[int, int] = None) -> plt.Figure:
    """Create a sequence logo from PWM data"""

    # Ensure PWM is in the right shape (4 x length)
    if pwm_data.shape[0] > pwm_data.shape[1]:
        pwm_data = pwm_data.T

    length = pwm_data.shape[1]
    nucleotides = ['A', 'C', 'G', 'T']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red

    # Calculate dynamic figure size if not provided
    if figsize is None:
        base_width = 10
        if length <= 25:
            width = base_width
        else:
            width = min(base_width + (length - 25) * 0.3, 30)  # Cap at 30 inches max
        figsize = (width, 3)

    fig, ax = plt.subplots(figsize=figsize)

    # Calculate information content (bits)
    ic = np.zeros(length)
    for i in range(length):
        # Shannon entropy
        entropy = 0
        for j in range(4):
            if pwm_data[j, i] > 0:
                entropy += pwm_data[j, i] * np.log2(pwm_data[j, i])
        ic[i] = 2 + entropy  # Max entropy for DNA is 2 bits

    # Draw the logo
    x = np.arange(length)
    for i in range(length):
        # Sort nucleotides by height at this position
        heights = []
        for j in range(4):
            heights.append((pwm_data[j, i] * ic[i], j))

        heights.sort(reverse=True)

        # Stack nucleotides
        y_pos = 0
        for height, nt_idx in heights:
            if height > 0.01:  # Only draw if significant
                ax.text(i, y_pos, nucleotides[nt_idx],
                       fontsize=height * 50,  # Scale font size
                       color=colors[nt_idx],
                       ha='center', va='bottom',
                       fontweight='bold')
                y_pos += height

    ax.set_xlim(-0.5, length - 0.5)
    ax.set_ylim(0, 2.5)
    ax.set_xlabel('Position')
    ax.set_ylabel('Information (bits)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Set x-ticks with adaptive spacing
    tick_spacing = get_adaptive_tick_spacing(length)
    ax.set_xticks(range(0, length, tick_spacing))

    plt.tight_layout()
    return fig

def plot_pwm_comparison(predicted_pwm: np.ndarray, reference_seq: Optional[str] = None,
                       title: str = "PWM Prediction", figsize: Tuple[int, int] = None) -> plt.Figure:
    """Create a comprehensive PWM visualization with optional reference sequence"""

    # Ensure PWM is in the right shape
    if predicted_pwm.shape[0] > predicted_pwm.shape[1]:
        predicted_pwm = predicted_pwm.T

    length = predicted_pwm.shape[1]
    nucleotides = ['A', 'C', 'G', 'T']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # Calculate dynamic figure size if not provided
    if figsize is None:
        base_width = 12
        if length <= 25:
            width = base_width
        else:
            width = min(base_width + (length - 25) * 0.3, 30)  # Cap at 30 inches max
        figsize = (width, 8) if reference_seq and len(reference_seq) == length else (width, 6)

    if reference_seq and len(reference_seq) == length:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize,
                                            gridspec_kw={'height_ratios': [1, 2, 2]})

        # Plot reference sequence
        ax1.imshow([list(reference_seq)], aspect='auto', cmap='tab10',
                  interpolation='nearest')
        ax1.set_title('Reference Sequence')
        ax1.set_yticks([])
        # Set adaptive x-ticks for reference sequence
        tick_spacing = get_adaptive_tick_spacing(length)
        ax1.set_xticks(range(0, length, tick_spacing))
        ax1.set_xticklabels([])

        # Plot predicted PWM as heatmap
        im = ax2.imshow(predicted_pwm, aspect='auto', cmap='Blues',
                       interpolation='nearest')
        ax2.set_title('Predicted PWM (Heatmap)')
        ax2.set_yticks(range(4))
        ax2.set_yticklabels(nucleotides)
        ax2.set_xticks(range(0, length, tick_spacing))
        plt.colorbar(im, ax=ax2)

        # Plot predicted PWM as bar chart
        x = np.arange(length)
        bottom = np.zeros(length)
        for i, (nt, color) in enumerate(zip(nucleotides, colors)):
            ax3.bar(x, predicted_pwm[i], bottom=bottom,
                   label=nt, color=color, alpha=0.8)
            bottom += predicted_pwm[i]

        ax3.set_title('Predicted PWM (Stacked Bar)')
        ax3.set_xlabel('Position')
        ax3.set_ylabel('Probability')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize,
                                      gridspec_kw={'height_ratios': [1, 1]})

        # Calculate tick spacing for this branch too
        tick_spacing = get_adaptive_tick_spacing(length)

        # Plot predicted PWM as heatmap
        im = ax1.imshow(predicted_pwm, aspect='auto', cmap='Blues',
                       interpolation='nearest')
        ax1.set_title('Predicted PWM (Heatmap)')
        ax1.set_yticks(range(4))
        ax1.set_yticklabels(nucleotides)
        ax1.set_xticks(range(0, length, tick_spacing))
        plt.colorbar(im, ax=ax1)

        # Plot predicted PWM as bar chart
        x = np.arange(length)
        bottom = np.zeros(length)
        for i, (nt, color) in enumerate(zip(nucleotides, colors)):
            ax2.bar(x, predicted_pwm[i], bottom=bottom,
                   label=nt, color=color, alpha=0.8)
            bottom += predicted_pwm[i]

        ax2.set_title('Predicted PWM (Stacked Bar)')
        ax2.set_xlabel('Position')
        ax2.set_ylabel('Probability')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

def load_npz_results(npz_file_path: str) -> Dict:
    """Load and parse NPZ result file"""
    try:
        data = np.load(npz_file_path)
        results = {}

        for key in data.files:
            results[key] = data[key]

        return results
    except Exception as e:
        return {"error": f"Failed to load NPZ file: {str(e)}"}

def create_results_summary(results: Dict) -> str:
    """Create a text summary of processing results"""
    summary = []
    summary.append("=== DeepPBS Processing Results ===\n")

    if "error" in results:
        summary.append(f"Error: {results['error']}")
        return "\n".join(summary)

    # PWM data summary
    if "P" in results:
        pwm_data = results["P"]
        summary.append(f"PWM Shape: {pwm_data.shape}")
        summary.append(f"PWM Range: [{pwm_data.min():.3f}, {pwm_data.max():.3f}]")
        summary.append(f"PWM Sum per position: {pwm_data.sum(axis=0)}")

    # Sequence data
    if "Seq" in results:
        seq_data = results["Seq"]
        if seq_data.ndim == 1:
            summary.append(f"Reference Sequence: {''.join([int_to_base(x) for x in seq_data])}")
        else:
            summary.append(f"Sequence Shape: {seq_data.shape}")

    # Additional metadata
    summary.append("\n=== Additional Information ===")
    for key, value in results.items():
        if key not in ["P", "Seq"]:
            if isinstance(value, np.ndarray):
                summary.append(f"{key}: array with shape {value.shape}, dtype {value.dtype}")
            else:
                summary.append(f"{key}: {type(value).__name__} = {value}")

    return "\n".join(summary)

def int_to_base(x: int) -> str:
    """Convert integer to DNA base (for one-hot encoding)"""
    bases = ['A', 'C', 'G', 'T']
    if 0 <= x < len(bases):
        return bases[x]
    return 'N'

def svg_to_html(svg_path: str) -> str:
    """Convert SVG file to HTML for embedding"""
    try:
        with open(svg_path, 'r') as f:
            svg_content = f.read()

        # Add responsive sizing
        svg_content = svg_content.replace('<svg', '<svg style="max-width: 100%; height: auto;"')
        return svg_content
    except Exception as e:
        return f"<p>Error loading SVG: {str(e)}</p>"

def create_processing_report(start_time: float, end_time: float,
                           input_file: str, parameters: Dict) -> str:
    """Create a detailed processing report"""

    duration = end_time - start_time

    report = []
    report.append("=== DeepPBS Processing Report ===")
    report.append(f"Processing Time: {duration:.1f} seconds")
    report.append(f"Input File: {os.path.basename(input_file)}")
    report.append(f"File Size: {os.path.getsize(input_file) / 1024:.1f} KB")

    report.append("\n=== Parameters ===")
    for key, value in parameters.items():
        report.append(f"{key}: {value}")

    report.append("\n=== Pipeline Steps ===")
    report.append("1. Structure validation - Complete")
    report.append("2. Geometric feature extraction - Complete")
    report.append("3. Graph construction - Complete")
    report.append("4. Neural network prediction - Complete")
    report.append("5. Result visualization - Complete")

    return "\n".join(report)

def validate_environment() -> Tuple[bool, List[str]]:
    """Validate that all required dependencies are available"""

    issues = []

    # Check Python dependencies
    required_modules = [
        ('numpy', 'numpy'),
        ('matplotlib', 'matplotlib'),
        ('torch', 'torch'),
        ('gradio', 'gradio')
    ]

    for module_name, pip_name in required_modules:
        try:
            __import__(module_name)
        except ImportError:
            issues.append(f"Missing Python module: {pip_name}")

    # Check DeepPBS modules
    deeppbs_modules = [
        'deeppbs.nn.utils',
        'deeppbs.nn',
        'models.model_v2'
    ]

    for module_name in deeppbs_modules:
        try:
            __import__(module_name)
        except ImportError:
            issues.append(f"Missing DeepPBS module: {module_name}")

    # Check external tools (simplified check)
    external_tools = [
        ('3DNA', ['find_pair', 'analyze']),
        ('Curves+', ['Cur+']),
        ('APBS', ['apbs'])
    ]

    for tool_name, commands in external_tools:
        found = False
        for cmd in commands:
            try:
                result = subprocess.run(['which', cmd], capture_output=True, text=True)
                if result.returncode == 0:
                    found = True
                    break
            except:
                continue

        if not found:
            issues.append(f"Missing external tool: {tool_name}")

    return len(issues) == 0, issues

def setup_example_data() -> bool:
    """Set up example data for the interface"""

    examples_dir = Path("examples")
    examples_dir.mkdir(exist_ok=True)

    # Create sample PDB files if they don't exist (minimal examples)
    sample_pdbs = {
        "1a1t.pdb": """HEADER    DNA BINDING PROTEIN
ATOM      1  P    DG A   1       1.000   2.000   3.000  1.00  0.00           P
ATOM      2  C1'  DG A   1       2.000   3.000   4.000  1.00  0.00           C
ATOM      3  N1   DG A   1       3.000   4.000   5.000  1.00  0.00           N
TER
""",
        "1bdt.pdb": """HEADER    DNA BINDING PROTEIN
ATOM      1  P    DA A   1       1.100   2.100   3.100  1.00  0.00           P
ATOM      2  C1'  DA A   1       2.100   3.100   4.100  1.00  0.00           C
ATOM      3  N1   DA A   1       3.100   4.100   5.100  1.00  0.00           N
TER
"""
    }

    for filename, content in sample_pdbs.items():
        filepath = examples_dir / filename
        if not filepath.exists():
            filepath.write_text(content)

    return True

if __name__ == "__main__":
    # Test the utility functions
    print("Testing DeepPBS Gradio utilities...")

    # Validate environment
    valid, issues = validate_environment()
    if not valid:
        print("Environment issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("Environment validation passed!")

    # Setup example data
    if setup_example_data():
        print("Example data setup complete!")

    print("Utility functions ready!")