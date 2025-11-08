#!/bin/bash

# Zyntra Backend Startup Script

echo "🚀 Starting Zyntra Backend API..."
echo ""

# Activate virtual environment
echo "📦 Activating virtual environment..."
source zyntra_venv/bin/activate

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found!"
    echo "📄 Creating .env from .env.example..."
    cp .env.example .env
    echo "✅ .env file created. Please update it with your configuration."
    echo ""
fi

# Start the server
echo "🌐 Starting FastAPI server..."
echo "📍 API will be available at: http://localhost:8000"
echo "📚 Swagger Docs: http://localhost:8000/docs"
echo "📖 ReDoc: http://localhost:8000/redoc"
echo ""

python main.py
