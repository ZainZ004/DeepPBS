#!/usr/bin/env python3
"""
Gradio Web Interface for DeepPBS
A user-friendly web interface for protein-DNA binding specificity prediction
"""

import gradio as gr
import os
import sys
import tempfile
import shutil
import subprocess
import json
import time
import re
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

# Add the run directory to Python path for importing DeepPBS modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'run'))

# Import utility functions
from gradio_utils import (
    create_sequence_logo, plot_pwm_comparison, load_npz_results,
    create_results_summary, svg_to_html, create_processing_report,
    validate_environment, setup_example_data
)

class DeepPBSGradioInterface:
    def __init__(self):
        self.temp_base_dir = tempfile.mkdtemp(prefix="deeppbs_gradio_")
        self.current_session = None
        self.supported_formats = ['.pdb', '.cif', '.ent']
        self.max_file_size = 50 * 1024 * 1024  # 50MB limit

        # Setup environment for external tools
        self.setup_environment()

    def setup_environment(self):
        """Setup environment variables for external tools"""
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))

            # Setup paths for external tools
            dependencies_bin = os.path.join(script_dir, "dependencies", "bin")
            x3dna_path = os.path.join(script_dir, "x3dna-v2.3-linux-64bit", "x3dna-v2.3")

            # Add to PATH
            current_path = os.environ.get("PATH", "")
            if dependencies_bin not in current_path:
                os.environ["PATH"] = f"{dependencies_bin}:{current_path}"

            # Set X3DNA environment variable
            if os.path.exists(x3dna_path):
                os.environ["X3DNA"] = x3dna_path

            print(f"✓ Environment setup complete")
            print(f"  - Dependencies bin: {dependencies_bin}")
            print(f"  - X3DNA path: {x3dna_path}")

        except Exception as e:
            print(f"⚠️  Environment setup failed: {e}")
            print("  External tools may not be available")

    def validate_structure_file(self, file_path: str) -> Tuple[bool, str]:
        """Validate PDB/mmCIF structure file"""
        if not file_path or not os.path.exists(file_path):
            return False, "File not found"

        file_size = os.path.getsize(file_path)
        if file_size > self.max_file_size:
            return False, f"File too large ({file_size / 1024 / 1024:.1f}MB > 50MB limit)"

        file_ext = Path(file_path).suffix.lower()
        if file_ext not in self.supported_formats:
            return False, f"Unsupported format {file_ext}. Supported: {', '.join(self.supported_formats)}"

        # Basic content validation
        try:
            with open(file_path, 'r') as f:
                first_lines = [f.readline().strip() for _ in range(10)]

            if file_ext == '.pdb':
                # Check for PDB format indicators
                if not any(line.startswith(('ATOM', 'HETATM', 'HEADER')) for line in first_lines):
                    return False, "Invalid PDB format"
            elif file_ext == '.cif':
                # Check for mmCIF format indicators
                if not any('data_' in line or '_atom_site.' in line for line in first_lines):
                    return False, "Invalid mmCIF format"

            return True, "Valid structure file"

        except Exception as e:
            return False, f"Error reading file: {str(e)}"

    def validate_pwm_file(self, file_path: Optional[str]) -> Tuple[bool, str]:
        """Validate PWM alignment file"""
        if not file_path:
            return True, "No PWM file provided (optional)"

        if not os.path.exists(file_path):
            return False, "PWM file not found"

        try:
            with open(file_path, 'r') as f:
                content = f.read().strip()

            # Basic PWM format check (JASPAR-like)
            lines = content.split('\n')
            if len(lines) < 4:
                return False, "PWM file too short"

            # Check for nucleotide counts
            valid_nucleotides = ['A', 'C', 'G', 'T']
            header_line = lines[0].strip()

            # Simple validation - could be enhanced
            if not any(nt in header_line.upper() for nt in valid_nucleotides):
                # Check if subsequent lines contain nucleotide data
                data_found = False
                for line in lines[1:5]:
                    if any(nt in line.upper() for nt in valid_nucleotides):
                        data_found = True
                        break
                if not data_found:
                    return False, "Invalid PWM format - no nucleotide data found"

            return True, "Valid PWM file"

        except Exception as e:
            return False, f"Error reading PWM file: {str(e)}"

    def create_session_directory(self) -> str:
        """Create a unique session directory"""
        timestamp = int(time.time())
        session_dir = os.path.join(self.temp_base_dir, f"session_{timestamp}")
        os.makedirs(session_dir, exist_ok=True)

        # Create subdirectories matching DeepPBS expectations
        os.makedirs(os.path.join(session_dir, "pdb"), exist_ok=True)
        os.makedirs(os.path.join(session_dir, "npz"), exist_ok=True)
        os.makedirs(os.path.join(session_dir, "output"), exist_ok=True)
        os.makedirs(os.path.join(session_dir, "output", "npzs"), exist_ok=True)

        self.current_session = session_dir
        return session_dir

    def prepare_processing_environment(self, session_dir: str, structure_file: str,
                                     pwm_file: Optional[str] = None) -> Dict[str, str]:
        """Prepare files and configuration for processing"""

        # Copy structure file to session pdb directory
        pdb_dir = os.path.join(session_dir, "pdb")
        structure_filename = os.path.basename(structure_file)
        session_structure_path = os.path.join(pdb_dir, structure_filename)
        shutil.copy2(structure_file, session_structure_path)

        # Create input.txt file
        input_txt_path = os.path.join(session_dir, "input.txt")
        with open(input_txt_path, 'w') as f:
            if pwm_file:
                pwm_basename = os.path.basename(pwm_file)
                f.write(f"{structure_filename},{pwm_basename}\n")
            else:
                f.write(f"{structure_filename}\n")

        # Copy PWM file if provided
        if pwm_file:
            session_pwm_path = os.path.join(pdb_dir, os.path.basename(pwm_file))
            shutil.copy2(pwm_file, session_pwm_path)

        # Create processing configuration
        process_config = {
            "PDB_FILES_PATH": pdb_dir,
            "FEATURE_DATA_PATH": os.path.join(session_dir, "npz")
        }

        process_config_path = os.path.join(session_dir, "process_config.json")
        with open(process_config_path, 'w') as f:
            json.dump(process_config, f, indent=2)

        return {
            "session_dir": session_dir,
            "input_txt": input_txt_path,
            "process_config": process_config_path,
            "pdb_dir": pdb_dir,
            "npz_dir": os.path.join(session_dir, "npz"),
            "output_dir": os.path.join(session_dir, "output")
        }

    def run_processing(self, paths: Dict[str, str], progress_callback) -> Tuple[bool, str]:
        """Run the DeepPBS processing pipeline"""

        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))

            # Step 1: Run preprocessing with environment setup
            progress_callback(0.1, "Starting preprocessing...")

            # Setup script paths
            proc_source_script = os.path.join(script_dir, "run", "process", "proc_source.sh")
            process_script = os.path.join(script_dir, "run", "process_co_crystal.py")

            # Create a shell script that sources environment and runs the processing
            process_cmd = f"""#!/bin/bash
source {proc_source_script}
{sys.executable} {process_script} {paths["input_txt"]} {paths["process_config"]} --no_pwm
"""

            process_script_path = os.path.join(paths["session_dir"], "run_preprocess.sh")
            with open(process_script_path, 'w') as f:
                f.write(process_cmd)
            os.chmod(process_script_path, 0o755)

            result = subprocess.run(['/bin/bash', process_script_path], capture_output=True, text=True, cwd=paths["session_dir"])

            if result.returncode != 0:
                error_msg = "Preprocessing failed"
                if result.stderr:
                    # Extract meaningful error information
                    stderr_lines = result.stderr.strip().split('\n')
                    if stderr_lines:
                        # Take the last non-empty line as the most relevant error
                        for line in reversed(stderr_lines):
                            if line.strip() and not line.startswith('ERROR:'):
                                error_msg = f"Preprocessing failed: {line.strip()}"
                                break
                        else:
                            # If no meaningful line found, use the last error line
                            for line in reversed(stderr_lines):
                                if line.strip():
                                    error_msg = f"Preprocessing failed: {line.strip()}"
                                    break
                elif result.stdout:
                    stdout_lines = result.stdout.strip().split('\n')
                    if stdout_lines:
                        # Take the last meaningful output line
                        for line in reversed(stdout_lines):
                            if line.strip() and not line.startswith('ERROR:'):
                                error_msg = f"Preprocessing failed: {line.strip()}"
                                break
                        else:
                            for line in reversed(stdout_lines):
                                if line.strip():
                                    error_msg = f"Preprocessing failed: {line.strip()}"
                                    break
                return False, error_msg

            progress_callback(0.4, "Preprocessing complete. Starting prediction...")

            # Step 2: Create prediction input
            predict_input = os.path.join(paths["session_dir"], "predict_input.txt")
            npz_files = [f for f in os.listdir(paths["npz_dir"]) if f.endswith('.npz')]

            if not npz_files:
                return False, "No feature files generated during preprocessing"

            with open(predict_input, 'w') as f:
                for npz_file in npz_files:
                    f.write(npz_file + '\n')

            # Step 3: Run prediction with environment setup
            predict_script = os.path.join(script_dir, "run", "predict.py")
            pred_config = os.path.join(script_dir, "run", "process", "pred_configs", "pred_config_deeppbs.json")

            predict_cmd = f"""#!/bin/bash
source {proc_source_script}
{sys.executable} {predict_script} {predict_input} {paths["output_dir"]} -c {pred_config}
"""

            predict_script_path = os.path.join(paths["session_dir"], "run_predict.sh")
            with open(predict_script_path, 'w') as f:
                f.write(predict_cmd)
            os.chmod(predict_script_path, 0o755)

            result = subprocess.run(['/bin/bash', predict_script_path], capture_output=True, text=True, cwd=paths["session_dir"])

            if result.returncode != 0:
                error_msg = "Prediction failed"
                if result.stderr:
                    # Extract meaningful error information
                    stderr_lines = result.stderr.strip().split('\n')
                    if stderr_lines:
                        for line in reversed(stderr_lines):
                            if line.strip() and not line.startswith('ERROR:'):
                                error_msg = f"Prediction failed: {line.strip()}"
                                break
                        else:
                            for line in reversed(stderr_lines):
                                if line.strip():
                                    error_msg = f"Prediction failed: {line.strip()}"
                                    break
                elif result.stdout:
                    stdout_lines = result.stdout.strip().split('\n')
                    if stdout_lines:
                        for line in reversed(stdout_lines):
                            if line.strip() and not line.startswith('ERROR:'):
                                error_msg = f"Prediction failed: {line.strip()}"
                                break
                        else:
                            for line in reversed(stdout_lines):
                                if line.strip():
                                    error_msg = f"Prediction failed: {line.strip()}"
                                    break
                return False, error_msg

            progress_callback(0.9, "Prediction complete. Preparing results...")

            return True, "Processing completed successfully"

        except Exception as e:
            return False, f"Processing error: {str(e)}"

    def collect_results(self, paths: Dict[str, str]) -> Dict:
        """Collect and organize processing results"""
        output_dir = paths["output_dir"]
        results = {
            "svg_files": [],
            "npz_files": [],
            "success": False,
            "message": ""
        }

        try:
            # Find SVG files
            svg_files = [f for f in os.listdir(output_dir) if f.endswith('.svg')]
            for svg_file in svg_files:
                results["svg_files"].append(os.path.join(output_dir, svg_file))

            # Find NPZ files
            npz_dir = os.path.join(output_dir, "npzs")
            if os.path.exists(npz_dir):
                npz_files = [f for f in os.listdir(npz_dir) if f.endswith('.npz')]
                for npz_file in npz_files:
                    results["npz_files"].append(os.path.join(npz_dir, npz_file))

            if results["svg_files"] or results["npz_files"]:
                results["success"] = True
                results["message"] = f"Generated {len(results['svg_files'])} plots and {len(results['npz_files'])} data files"
            else:
                results["message"] = "No output files generated"

        except Exception as e:
            results["message"] = f"Error collecting results: {str(e)}"

        return results

    def process_structure(self, structure_file, pwm_file=None, model_type="DeepPBS"):
        """Main processing function for Gradio interface"""
        start_time = time.time()

        # Validate inputs
        valid, message = self.validate_structure_file(structure_file.name if structure_file else "")
        if not valid:
            error_report = f"Validation failed: {message}"
            return None, None, error_report

        if pwm_file:
            valid, message = self.validate_pwm_file(pwm_file.name)
            if not valid:
                error_report = f"PWM validation failed: {message}"
                return None, None, error_report

        # Create session and prepare environment
        session_dir = self.create_session_directory()

        try:
            # Progress tracking
            progress = gr.Progress()
            progress(0, "Preparing processing environment...")

            # Prepare files and configuration
            paths = self.prepare_processing_environment(
                session_dir,
                structure_file.name,
                pwm_file.name if pwm_file else None
            )

            # Run processing
            success, message = self.run_processing(paths, progress)

            if not success:
                error_report = f"Processing failed: {message}"
                return None, None, error_report

            # Collect results
            progress(0.95, "Collecting results...")
            results = self.collect_results(paths)

            if not results["success"]:
                error_report = f"No results generated: {results['message']}"
                return None, None, error_report

            progress(1.0, "Creating visualizations...")

            # Load NPZ results for advanced visualization
            npz_visualizations = []
            if results["npz_files"]:
                for npz_file in results["npz_files"]:
                    try:
                        npz_data = load_npz_results(npz_file)
                        if "P" in npz_data and not "error" in npz_data:
                            # Create sequence logo
                            pwm_data = npz_data["P"]
                            logo_fig = create_sequence_logo(pwm_data, "Predicted Binding Specificity")
                            npz_visualizations.append(logo_fig)
                    except Exception as e:
                        print(f"Warning: Could not create visualization from {npz_file}: {e}")
                        continue

            # Prepare display data
            plots = []
            for svg_file in results["svg_files"]:
                plots.append(svg_file)

            # Add our custom visualizations if available
            if npz_visualizations:
                plots.extend(npz_visualizations)

            # Prepare download data
            downloads = []
            for npz_file in results["npz_files"]:
                downloads.append(npz_file)
            for svg_file in results["svg_files"]:
                downloads.append(svg_file)

            # Create processing report
            end_time = time.time()
            parameters = {
                "Model": model_type,
                "Structure File": os.path.basename(structure_file.name),
                "PWM File": os.path.basename(pwm_file.name) if pwm_file else "None",
                "Skip PWM": "No",
                "Skip Cleaning": "No"
            }

            report = create_processing_report(start_time, end_time, structure_file.name, parameters)

            return (
                gr.update(value=plots[0] if plots else None),  # Display first plot
                gr.update(value=downloads),  # Files for download
                gr.update(value=report)  # Status message
            )

        except Exception as e:
            error_report = f"Unexpected error: {str(e)}"
            return None, None, error_report

        finally:
            # Cleanup old sessions to prevent disk space issues
            self.cleanup_old_sessions()

    def cleanup_session(self):
        """Clean up temporary files"""
        if self.current_session and os.path.exists(self.current_session):
            shutil.rmtree(self.current_session, ignore_errors=True)
            self.current_session = None

    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """Clean up old session directories to prevent disk space issues"""
        try:
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600

            for item in os.listdir(self.temp_base_dir):
                item_path = os.path.join(self.temp_base_dir, item)
                if os.path.isdir(item_path) and item.startswith("session_"):
                    # Check if directory is older than max_age_hours
                    dir_stat = os.stat(item_path)
                    dir_age = current_time - dir_stat.st_mtime

                    if dir_age > max_age_seconds:
                        shutil.rmtree(item_path, ignore_errors=True)

        except Exception:
            # Silently ignore cleanup errors
            pass

def create_interface():
    """Create and return the Gradio interface"""

    processor = DeepPBSGradioInterface()

    # Set up example data
    setup_example_data()

    # Validate environment
    valid, issues = validate_environment()
    if not valid:
        print("Environment validation issues found:")
        for issue in issues:
            print(f"  - {issue}")

    # Create interface with Gradio 6.0+ compatible syntax
    with gr.Blocks(title="DeepPBS - Protein-DNA Binding Specificity Prediction") as interface:

        # Add custom CSS using HTML component
        gr.HTML("""
        <style>
        .gradio-container {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
        .title {
            text-align: center;
            color: #2c3e50;
            margin-bottom: 2rem;
        }
        .info-box {
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 5px;
            padding: 15px;
            margin: 10px 0;
        }
        </style>
        """)

        gr.HTML("""
        <div class="title">
            <h1>🧬 DeepPBS</h1>
            <h3>Geometric Deep Learning for Protein-DNA Binding Specificity Prediction</h3>
            <p>Upload a protein-DNA complex structure to predict DNA binding preferences</p>
        </div>
        """)

        if not valid:
            gr.HTML(f"""
            <div class="info-box" style="background-color: #fff3cd; border-color: #ffeaa7; color: #856404;">
                <strong>⚠️ Environment Issues Detected:</strong><br>
                {'<br>'.join(issues)}
            </div>
            """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📁 Input Files")

                structure_file = gr.File(
                    label="Structure File (PDB/mmCIF)",
                    file_types=[".pdb", ".cif", ".ent"],
                    type="filepath"
                )

                pwm_file = gr.File(
                    label="PWM Alignment File (Optional)",
                    file_types=[".pwm", ".txt"],
                    type="filepath"
                )

                gr.Markdown("### ⚙️ Configuration")

                model_type = gr.Dropdown(
                    choices=["DeepPBS", "DeepPBSwithDNAseqInfo", "BaseReadout", "ShapeReadout"],
                    value="DeepPBS",
                    label="Prediction Model"
                )

                with gr.Row():
                    skip_pwm = gr.Checkbox(label="Skip PWM alignment", value=False)
                    skip_clean = gr.Checkbox(label="Skip structure cleaning", value=False)

                process_btn = gr.Button("🚀 Process Structure", variant="primary")

                with gr.Accordion("ℹ️ Processing Information", open=False):
                    info_text = gr.Textbox(
                        label="Status & Details",
                        interactive=False,
                        lines=8,
                        placeholder="Ready to process..."
                    )

            with gr.Column(scale=2):
                gr.Markdown("### 📊 Results")

                with gr.Tab("📈 Visualization"):
                    plot_output = gr.Plot(label="PWM Prediction")

                with gr.Tab("🎯 Sequence Logo"):
                    logo_output = gr.Image(label="Sequence Logo")

                with gr.Tab("💾 Download Results"):
                    download_files = gr.File(
                        label="Result Files",
                        file_count="multiple"
                    )

                with gr.Tab("📋 Summary"):
                    summary_text = gr.Textbox(
                        label="Processing Summary",
                        interactive=False,
                        lines=12
                    )

        # Event handlers
        process_btn.click(
            fn=processor.process_structure,
            inputs=[structure_file, pwm_file, model_type],
            outputs=[plot_output, download_files, info_text]
        )

        # Examples
        gr.Examples(
            examples=[
                ["examples/1a1t.pdb", None, "DeepPBS"],
                ["examples/1bdt.pdb", None, "DeepPBS"],
            ],
            inputs=[structure_file, pwm_file, model_type],
            outputs=[plot_output, download_files, info_text],
            fn=processor.process_structure,
            cache_examples=False,
            label="Example Structures"
        )

        gr.Markdown("""
        ### 📖 Instructions
        1. **Upload Structure**: Select a protein-DNA complex structure file (PDB or mmCIF format)
        2. **Optional PWM**: Optionally upload a PWM file for alignment reference
        3. **Choose Model**: Select the prediction model (DeepPBS recommended for most cases)
        4. **Process**: Click "Process Structure" to start the analysis
        5. **View Results**: Explore results in different tabs and download files

        ### ⏱️ Processing Information
        - **Typical Duration**: 1-3 minutes depending on structure size
        - **Pipeline Steps**: Structure validation → Geometric feature extraction → Graph construction → Neural network prediction → Result visualization
        - **External Tools**: Uses 3DNA, Curves+, and APBS for geometric analysis

        ### 🔧 Model Options
        - **DeepPBS**: Standard model using protein shape information (recommended)
        - **DeepPBSwithDNAseqInfo**: Enhanced model with DNA sequence information
        - **BaseReadout**: Basic model for base-specific readout
        - **ShapeReadout**: Model focusing on DNA shape readout

        ### 📁 Output Files
        - **SVG Plots**: Visualization of predicted PWMs and sequence logos
        - **NPZ Files**: Numerical prediction data for further analysis
        - **Processing Report**: Detailed information about the analysis pipeline
        """)

        gr.HTML("""
        <div style="text-align: center; margin-top: 2rem; padding: 1rem; background-color: #f8f9fa; border-radius: 5px;">
            <p><strong>DeepPBS</strong> - Geometric Deep Learning Framework for Protein-DNA Binding Specificity</p>
            <p style="font-size: 0.9em; color: #6c757d;">
                For command-line usage, see the original DeepPBS pipeline in <code>run/</code> directory
            </p>
        </div>
        """)

        return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False
    )