#!/bin/bash

# DeepPBS Gradio Web Interface Launcher
# This script sets up the environment and launches the Gradio web interface

set -e

echo "🧬 DeepPBS Gradio Web Interface Launcher"
echo "=========================================="

# Check if conda/mamba is available
if command -v mamba &> /dev/null; then
    CONDA_CMD="mamba"
elif command -v conda &> /dev/null; then
    CONDA_CMD="conda"
else
    echo "❌ Error: Neither mamba nor conda found. Please install conda or mamba first."
    exit 1
fi

echo "📦 Using $CONDA_CMD for environment management"

# Check if deeppbs environment exists
if $CONDA_CMD env list | grep -q "deeppbs"; then
    echo "✅ DeepPBS environment found"
else
    echo "❌ Error: DeepPBS environment not found. Please create it first:"
    echo "   $CONDA_CMD env create -f deeppbs.yaml"
    exit 1
fi

# Activate the environment
echo "🔄 Activating deeppbs environment..."
eval "$($CONDA_CMD shell.bash hook)"
$CONDA_CMD activate deeppbs

# Verify Gradio is installed
if python -c "import gradio" &> /dev/null; then
    echo "✅ Gradio is installed"
else
    echo "⚠️  Gradio not found. Installing..."
    pip install gradio>=5.8.0 pillow>=10.0.0
fi

# Check for example data
if [ ! -d "examples" ]; then
    echo "📁 Creating example data directory..."
    mkdir -p examples
fi

# Set up environment variables for external tools
if [ -f "dependencies/bin/x3dna_setup" ]; then
    echo "🧪 Setting up 3DNA environment..."
    source dependencies/bin/x3dna_setup
fi

# Check for required directories
echo "📂 Checking directory structure..."
mkdir -p run/process/pdb
mkdir -p run/process/npz
mkdir -p run/process/output

echo ""
echo "🚀 Starting DeepPBS Gradio Web Interface..."
echo "   The interface will be available at: http://localhost:7860"
echo "   Press Ctrl+C to stop the server"
echo ""

# Launch the Gradio interface
python gradio_app.py "$@"