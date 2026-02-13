#!/bin/bash

# Exit on error
set -e

echo "Setting up GraphiteAI environment..."

# 1. Check for Python 3
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 could not be found."
    exit 1
fi

# 2. Create Virtual Environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# 3. Activate Virtual Environment
source venv/bin/activate

# 4. Install Dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 5. Create directories if they don't exist (redundant but safe)
mkdir -p backend/model training frontend images saved_models

echo "Setup complete! To run the app:"
echo "  source venv/bin/activate"
echo "  cd backend && python app.py"
