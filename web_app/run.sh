#!/bin/bash

# Fall Detection Web App Startup Script

echo "================================="
echo "Fall Detection Web App Launcher"
echo "================================="
echo ""

# Check if we're in the correct directory
if [ ! -f "app.py" ]; then
    echo "Error: app.py not found. Please run this script from the web_app directory."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "../venv" ] && [ ! -d "../.venv" ]; then
    echo "Warning: No virtual environment found."
    echo "It's recommended to use a virtual environment."
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check Python dependencies
echo "Checking dependencies..."
python3 -c "import flask" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Flask not found. Installing web app dependencies..."
    pip install -r requirements.txt
fi

# Check main project dependencies
python3 -c "import torch" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "PyTorch not found. Installing main project dependencies..."
    pip install -r ../requirements.txt
fi

# Check OpenPose models
if [ ! -f "../models/opencv_pose/pose_iter_440000.caffemodel" ]; then
    echo ""
    echo "Warning: OpenPose models not found."
    echo "Downloading models... (this may take a few minutes)"
    python3 ../scripts/download_pose_models.py
fi

# Check if at least one detection model exists
if [ ! -f "../models/improved_hybrid_detector.pt" ] && \
   [ ! -f "../models/checkpoints/hybrid_fall_detector.pt" ]; then
    echo ""
    echo "Warning: No trained models found."
    echo "The ensemble detector (rule-based) will still work, but neural models won't be available."
    echo "To train models, see the training scripts in ../scripts/"
    echo ""
fi

echo ""
echo "Starting Fall Detection Web App..."
echo "================================="
echo "Open your browser and navigate to: http://127.0.0.1:5000"
echo "Press Ctrl+C to stop the server"
echo ""

# Run the Flask app
python3 app.py
