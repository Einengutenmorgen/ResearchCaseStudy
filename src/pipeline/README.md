# Conversation Pipeline

A modular pipeline for processing, analyzing, and generating conversations from social media data.

## Project Structure

```
/pipeline/
├── README.md                           # This file
├── run_pipeline.py                     # Main entry point
└── conversation_pipeline/              # Main package
    ├── core/                           # Core processing modules
    │   ├── pipeline.py                 # Main pipeline implementation
    │   ├── data_processor.py          # Data processing utilities
    │   ├── preprocess_data.py         # Data preprocessing
    │   ├── main_pipeline.py           # Pipeline orchestration
    │   └── pipeline_openai.py         # OpenAI integration
    ├── components/                     # Pipeline components
    │   ├── analyzers/                 # Analysis components
    │   │   ├── openai_analyzer.py     # OpenAI-based analysis
    │   │   ├── similarity_analyzer.py # Text similarity analysis
    │   │   └── rouge_evaluator.py     # ROUGE metrics evaluation
    │   ├── generators/                # Generation components
    │   │   ├── enhanced_neutral_generator.py
    │   │   └── post_regenerator.py    # Post regeneration
    │   ├── storage.py                 # Data storage management
    │   ├── builder.py                 # Conversation building
    │   ├── predictor.py              # Conversation prediction
    │   ├── reply_filter.py           # Reply filtering
    │   ├── root_fetcher.py           # Root tweet fetching
    │   └── persona_manager.py        # User persona management
    └── utils/                         # Utility modules
        ├── data_chunker.py           # Data chunking utilities
        ├── post_collector.py         # Post collection
        ├── prompt_formatter.py       # Prompt formatting
        └── read_personas.py          # Persona reading utilities

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
export OPENAI_API_KEY=your_api_key_here
```

3. Run the pipeline:
```bash
python run_pipeline.py
```

## Components

### Core

- **pipeline.py**: Main pipeline implementation that orchestrates the conversation processing workflow
- **data_processor.py**: Handles data loading and basic processing
- **preprocess_data.py**: Implements data preprocessing steps
- **main_pipeline.py**: Manages the overall pipeline execution
- **pipeline_openai.py**: Integrates OpenAI services into the pipeline

### Components

#### Analyzers
- **openai_analyzer.py**: Uses OpenAI models for text analysis
- **similarity_analyzer.py**: Implements text similarity metrics
- **rouge_evaluator.py**: Provides ROUGE evaluation metrics

#### Generators
- **enhanced_neutral_generator.py**: Generates neutral descriptions of conversations
- **post_regenerator.py**: Regenerates posts with specific characteristics

#### Other Components
- **storage.py**: Manages data persistence
- **builder.py**: Builds conversation structures
- **predictor.py**: Predicts conversation characteristics
- **reply_filter.py**: Filters relevant replies
- **root_fetcher.py**: Retrieves root tweets
- **persona_manager.py**: Manages user personas

### Utils
- **data_chunker.py**: Handles data chunking for efficient processing
- **post_collector.py**: Collects posts from various sources
- **prompt_formatter.py**: Formats prompts for AI models
- **read_personas.py**: Utilities for reading and managing personas

## Usage

The pipeline can be run with different configurations:

```python
from conversation_pipeline.core.pipeline import ConversationPipeline

# Initialize the pipeline
pipeline = ConversationPipeline(
    input_file="path/to/data.csv",
    output_dir="path/to/output"
)

# Run the pipeline
pipeline.run(max_chunks=1)  # Process one chunk for testing
```

## Output

The pipeline generates:
1. Processed conversations in JSON format
2. User personas
3. Analysis results
4. Performance metrics

Results are saved in the specified output directory with the following structure:
```
output_dir/
├── conversations/       # Processed conversations
├── personas/           # Generated user personas
├── analysis/           # Analysis results
└── metrics/           # Performance metrics
```

## Contributing

1. Follow the established project structure
2. Add appropriate tests for new components
3. Update documentation as needed
4. Follow Python best practices and PEP 8 guidelines

## License

[Your License Here]
