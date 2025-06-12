#!/bin/bash

# Kill any existing uvicorn processes
echo "Cleaning up existing processes..."
pkill -f "uvicorn.*main:app" || true
sleep 2

# Check if port 8000 is free
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null ; then
    echo "Port 8000 is busy. Killing processes using it..."
    lsof -ti:8000 | xargs kill -9 2>/dev/null || true
    sleep 2
fi

# Set environment variables to prevent OpenMP conflicts
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1

# Set Python path to include the current directory
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "Starting Booker API server..."
echo "Environment variables set:"
echo "  KMP_DUPLICATE_LIB_OK=$KMP_DUPLICATE_LIB_OK"
echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "  PYTHONPATH=$PYTHONPATH"

cd api && uvicorn main:app --reload --host 0.0.0.0 --port 8000 