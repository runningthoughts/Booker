#!/bin/bash

# Booker Setup Script
echo "🚀 Setting up Booker - RAG-based Book Q&A System"
echo "=================================================="

# Check if ragtagKey is set
if [ -z "$ragtagKey" ]; then
    echo "❌ Error: Environment variable 'ragtagKey' is not set"
    echo "Please set your OpenAI API key:"
    echo "export ragtagKey=\"your-openai-api-key-here\""
    exit 1
fi

echo "✅ OpenAI API key found"

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Download spaCy model
echo "🧠 Downloading spaCy model..."
python -m spacy download en_core_web_sm

# Setup frontend
echo "🎨 Setting up React frontend..."
cd ui
npm install
cd ..

echo "✅ Setup complete!"
echo ""
echo "📚 Next steps:"
echo "1. Add your PDF files to the data/ directory"
echo "2. Run: python -m booker.ingest_book"
echo "3. Start the backend: cd api && python main.py"
echo "4. Start the frontend: cd ui && npm run dev"
echo ""
echo "🌐 Then visit http://localhost:3000 to start asking questions!" 