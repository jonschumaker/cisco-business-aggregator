#!/bin/bash

# Setup script for Cisco Business Aggregator

echo "Setting up Cisco Business Aggregator..."

# Create virtual environment
echo "Creating virtual environment..."
python -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating necessary directories..."
python setup_directories.py

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file from .env.example..."
    cp .env.example .env
    echo "Please edit the .env file with your API keys and settings."
fi

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit the .env file with your API keys and settings"
echo "2. Run the application with: python main.py"
echo ""
echo "To activate the virtual environment in the future, run:"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "source venv/Scripts/activate"
else
    echo "source venv/bin/activate"
fi
