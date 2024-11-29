# Social Media Analysis Research Project

This project analyzes social media data and user personas using OpenAI's API.

## Project Structure
```
├── src/               # Source code
│   ├── pipeline/      # Core pipeline modules
│   └── utils/         # Utility functions
├── notebooks/         # Jupyter notebooks
├── data/             # Input data
├── Results/          # Analysis outputs
├── tests/            # Unit tests
└── requirements.txt  # Project dependencies
```

## Setup
1. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage
1. Data Processing Pipeline:
```bash
python src/pipeline/pipeline_openai.py
```

2. Rerun Descriptions:
```bash
python src/pipeline/rerun_descriptions.py
```

## Components
- `pipeline_openai.py`: Core pipeline for data processing and analysis
- `rerun_descriptions.py`: Script for reprocessing descriptions
- `compare_descriptions.py`: Comparison functionality
- `read_personas.py`: Persona reading functionality

## Development
- Use `pytest` for running tests
- Follow PEP 8 style guidelines
- Add type hints to new code
