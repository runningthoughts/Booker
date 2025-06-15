#!/bin/bash

# Booker startup script with OpenMP fix
export KMP_DUPLICATE_LIB_OK=TRUE

echo "Starting Booker with OpenMP compatibility fix..."

# Choose what to run
case "$1" in
    "api")
        echo "Starting API server..."
        uvicorn api.main:app --host 0.0.0.0 --port 8000
        ;;
    "viz")
        echo "Starting visualization app..."
        python -m booker.viz.cli
        ;;
    "both")
        echo "Starting both API server and viz app..."
        # Start API in background
        uvicorn api.main:app --host 0.0.0.0 --port 8000 &
        # Start viz app
        python -m booker.viz.cli
        ;;
    *)
        echo "Usage: $0 {api|viz|both}"
        echo "  api  - Start API server only"
        echo "  viz  - Start visualization app only"  
        echo "  both - Start both services"
        exit 1
        ;;
esac

# Booker Startup Script
echo "🚀 Starting Booker - RAG-based Book Q&A System"
echo "=============================================="

# Check if ragtagKey is set
if [ -z "$ragtagKey" ]; then
    echo "❌ Error: Environment variable 'ragtagKey' is not set"
    echo "Please set your OpenAI API key:"
    echo "export ragtagKey=\"your-openai-api-key-here\""
    exit 1
fi

echo "✅ OpenAI API key found"

# Function to check if port is in use
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null ; then
        echo "⚠️  Port $1 is already in use"
        return 1
    else
        return 0
    fi
}

# Check ports
if ! check_port 8000; then
    echo "Backend may already be running on port 8000"
fi

if ! check_port 3000; then
    echo "Frontend may already be running on port 3000"
fi

echo ""
echo "🔧 Starting Backend API..."
echo "Command: PYTHONPATH=. python api/main.py"
echo "Will run on: http://localhost:8000"
echo ""

# Start backend in background
PYTHONPATH=. python api/main.py &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 3

# Test backend
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ Backend started successfully!"
else
    echo "❌ Backend failed to start"
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

echo ""
echo "🎨 Starting Frontend..."
echo "Command: cd ui && npm run dev"
echo "Will run on: http://localhost:3000"
echo ""

# Start frontend
cd ui && npm run dev &
FRONTEND_PID=$!

echo ""
echo "🌟 Booker is starting up!"
echo ""
echo "📍 URLs:"
echo "   Frontend: http://localhost:3000"
echo "   Backend:  http://localhost:8000"
echo "   Health:   http://localhost:8000/health"
echo ""
echo "💡 To stop both services:"
echo "   Press Ctrl+C or run: kill $BACKEND_PID $FRONTEND_PID"
echo ""
echo "🎉 Happy questioning!"

# Wait for user to stop
wait 