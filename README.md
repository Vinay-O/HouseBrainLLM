# HouseBrain v1.1.0

AI-powered architectural design system that generates complete house designs using advanced layout algorithms and building code compliance validation.

## 🏠 Features

- **Multi-floor house design** with unlimited floor levels
- **Intelligent room layout** using architectural principles
- **Building code validation** with compliance scoring
- **2D floor plans** in SVG format
- **3D models** in OBJ format
- **Construction cost estimation**
- **Material requirements calculation**
- **REST API** for integration

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Demo Mode

```bash
# Generate sample house design
python -m api.main --demo
```

This will generate:
- `outputs/plan.json` → Complete house design in JSON format
- `outputs/level_0.svg` → Ground floor plan
- `outputs/level_1.svg` → First floor plan (if multi-story)
- `outputs/scene.obj` → 3D model for visualization

### 3. Start API Server

```bash
# Start the API server
python -m api.main --host 0.0.0.0 --port 8000
```

Visit `http://localhost:8000/docs` for interactive API documentation.

## 📋 API Usage

### Generate House Design

```python
import requests
import json

# Load sample input
with open("data/sample_input.json", "r") as f:
    house_input = json.load(f)

# Generate design
response = requests.post(
    "http://localhost:8000/design",
    json={"input": house_input}
)

result = response.json()
print(f"Design generated: {result['success']}")
print(f"Validation score: {result['validation']['compliance_score']}")
```

### Sample Input Format

```json
{
  "basicDetails": {
    "totalArea": 2800,
    "unit": "sqft",
    "floors": 2,
    "bedrooms": 3,
    "bathrooms": 3.5,
    "style": "Modern Contemporary",
    "budget": 8500000
  },
  "plot": {
    "length": 60,
    "width": 40,
    "unit": "ft",
    "orientation": "NE",
    "setbacks_ft": {
      "front": 5,
      "rear": 3,
      "left": 3,
      "right": 3
    }
  },
  "roomBreakdown": [
    {
      "type": "living_room",
      "count": 1,
      "size": "24' x 18'",
      "features": ["Fireplace", "Large windows", "Open to dining"]
    }
  ]
}
```

## 🏗️ Architecture

### Core Components

- **Schema** (`src/housebrain/schema.py`): Data models and validation
- **Layout Solver** (`src/housebrain/layout.py`): Room arrangement algorithms
- **Validator** (`src/housebrain/validate.py`): Building code compliance
- **LLM Interface** (`src/housebrain/llm.py`): AI reasoning integration
- **API Server** (`api/main.py`): FastAPI REST endpoints

### Design Process

1. **Input Parsing**: Validate user requirements and plot specifications
2. **Layout Generation**: Use grid-based algorithms for room placement
3. **AI Enhancement**: Apply architectural principles and optimization
4. **Validation**: Check building codes and design standards
5. **Rendering**: Generate 2D plans and 3D models
6. **Cost Estimation**: Calculate construction costs and materials

## 🎨 Room Types

The system supports comprehensive room types:

- **Living Spaces**: Living Room, Dining Room, Family Room
- **Bedrooms**: Master Bedroom, Bedroom, Study
- **Service Areas**: Kitchen, Bathroom, Half Bath, Utility
- **Storage**: Garage, Storage, Corridor
- **Circulation**: Entrance, Stairwell

## 🔍 Validation Features

- **Room Size Compliance**: Minimum area requirements
- **Stair Design**: Width, length, and headroom validation
- **Corridor Widths**: Accessibility standards
- **Daylight & Ventilation**: Window requirements for habitable rooms
- **Multi-floor Connectivity**: Stair connections between levels
- **Room Adjacency**: Logical placement of related spaces

## 🛠️ Development

### Project Structure

```
housebrain_v1_1/
├── api/
│   └── main.py              # FastAPI server
├── data/
│   └── sample_input.json    # Sample house requirements
├── src/
│   └── housebrain/
│       ├── __init__.py      # Package exports
│       ├── schema.py        # Data models
│       ├── layout.py        # Layout algorithms
│       ├── validate.py      # Validation logic
│       └── llm.py          # AI integration
├── outputs/                 # Generated files
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

### Adding New Features

1. **New Room Types**: Add to `RoomType` enum in `schema.py`
2. **Validation Rules**: Extend `HouseValidator` class in `validate.py`
3. **Layout Algorithms**: Enhance `LayoutSolver` in `layout.py`
4. **AI Integration**: Modify `HouseBrainLLM` in `llm.py`

## 🔮 Future Roadmap

### Phase 1: Residential Houses (Current)
- ✅ Basic house design generation
- ✅ Multi-floor support
- ✅ Building code validation
- ✅ 2D/3D rendering

### Phase 2: Indian Mixed-use
- 🚧 Residential + commercial buildings
- 🚧 Local building codes
- 🚧 Parking requirements

### Phase 3: Commercial Buildings
- 📋 Office buildings
- 📋 Retail spaces
- 📋 Industrial facilities

### Phase 4: High-rise & Apartments
- 📋 Multi-unit residential
- 📋 Vertical circulation
- 📋 Fire safety systems

### Phase 5: Global Expansion
- 📋 North American codes
- 📋 European standards
- 📋 Australian regulations

## 🤖 AI Integration

The system is designed for integration with local LLMs via Ollama:

```bash
# Install Ollama (if not already installed)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull DeepSeek R1 model
ollama pull deepseek-r1:8b

# Run with AI mode (future feature)
python -m api.main --ai-model deepseek-r1:8b

# Fine-tune with DeepSeek R1
python finetune_m2pro.py --dataset v3
```

## 📊 Performance

- **Design Generation**: ~2-5 seconds for typical houses
- **Validation**: Real-time compliance checking
- **Rendering**: SVG plans in <1 second, OBJ models in <2 seconds
- **API Response**: <10 seconds end-to-end

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure virtual environment is activated
2. **Missing Dependencies**: Run `pip install -r requirements.txt`
3. **SVG Generation**: Install `svgwrite` package
4. **Port Conflicts**: Use different port with `--port 8001`

### Debug Mode

```bash
# Run with verbose output
python -m api.main --demo --debug
```

## 📄 License

This project is part of the HouseBrain architectural AI system.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests and validation
5. Submit a pull request

## 📞 Support

For questions and support, please refer to the project documentation or create an issue in the repository.