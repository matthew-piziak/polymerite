{ pkgs ? import <nixpkgs> {} }:

let
  # Use Python 3.11 for better PyTorch compatibility (3.13 has typing issues)
  python = pkgs.python311;

  # Python with ML packages
  pythonEnv = python.withPackages (ps: with ps; [
    # Core scientific computing
    numpy
    pandas
    scipy
    scikit-learn

    # Data visualization
    matplotlib
    seaborn

    # Jupyter
    jupyter
    ipython
    notebook

    # Utilities
    pip
    virtualenv
    tqdm
    pyyaml
    requests
  ]);

in pkgs.mkShell {
  buildInputs = with pkgs; [
    pythonEnv

    # C++ standard library and compiler (required for PyTorch)
    stdenv.cc.cc.lib
    gcc
    glibc

    # System libraries for ML frameworks
    zlib
    libz

    # Git for version control
    git

    # Utilities
    tree
    htop
  ];

  shellHook = ''
    # Set library paths for PyTorch and other compiled packages
    export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [
      pkgs.stdenv.cc.cc.lib
      pkgs.zlib
      pkgs.glibc
    ]}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

    echo "========================================"
    echo "Polymerite ML Development Environment"
    echo "========================================"
    echo "Python version: $(python3 --version)"
    echo ""

    # Set up virtual environment for PyTorch and other packages
    export VENV_DIR=".venv"

    if [ ! -d "$VENV_DIR" ]; then
      echo "Creating virtual environment for ML packages..."
      echo "(PyTorch, PyTorch Geometric, Transformers)"
      echo ""
      # Create isolated venv without system site packages
      python3 -m venv $VENV_DIR --copies
      source $VENV_DIR/bin/activate

      # Upgrade pip and install compatible typing-extensions
      pip install --upgrade pip setuptools wheel
      pip install --upgrade typing-extensions

      echo ""
      echo "Virtual environment created at $VENV_DIR"
      echo ""
      echo "To install ML packages, run:"
      echo "  source .venv/bin/activate"
      echo "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
      echo "  pip install torch-geometric transformers pytorch-lightning"
      echo ""
      echo "For AMD GPU (ROCm), replace 'cpu' with 'rocm5.7'"
      echo "For NVIDIA GPU (CUDA), replace 'cpu' with 'cu118'"
      echo ""
    else
      echo "Virtual environment found at $VENV_DIR"
      echo "Activate with: source .venv/bin/activate"
      echo ""
    fi

    echo "Available scripts:"
    echo "  - python3 scripts/extract_point2_results.py    # Extract leaderboard data"
    echo "  - python3 scripts/download_point2_data.py      # Check dataset availability"
    echo "  - bash scripts/setup_ml_environment.sh         # Install ML dependencies"
    echo "  - bash scripts/test_installation.sh            # Test ML environment"
    echo ""
    echo "Quick start:"
    echo "  1. source .venv/bin/activate"
    echo "  2. bash scripts/setup_ml_environment.sh"
    echo "  3. bash scripts/test_installation.sh           # Verify installation"
    echo "========================================"
  '';

  # Environment variables
  NIX_SHELL_NAME = "polymerite";
}
