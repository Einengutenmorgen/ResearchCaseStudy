# Conversation Pipeline

A modular system for processing conversations and generating user personas from social media data.

## Project Structure

```
conversation_pipeline/
├── components/
│   ├── analyzers/
│   │   ├── openai_analyzer.py
│   │   └── rouge_evaluator.py
│   ├── persona/
│   │   ├── persona_creator.py
│   │   ├── post_analyzer.py
│   │   └── persona_pipeline.py
│   └── generators/
├── core/
│   ├── main_pipeline.py
│   ├── pipeline.py
│   └── data_processor.py
└── utils/
```

## Features

### Conversation Processing
- Data preprocessing and filtering
- Reply chain analysis
- Conversation reconstruction

### Persona Creation Pipeline
The persona creation pipeline is a new addition that operates independently from the conversation processing pipeline. It consists of three main components:

1. **PersonaCreator**
   - Generates detailed user personas using OpenAI's GPT-4
   - Includes caching mechanism for efficiency
   - Structured persona output with metadata

2. **PostAnalyzer**
   - Analyzes user posts for patterns and characteristics
   - Features:
     - Post metrics (word counts, lengths)
     - Temporal patterns (posting times, frequency)
     - Content patterns (URLs, emojis, hashtags)
     - Interaction patterns (mentions, engagement)
     - Language patterns (vocabulary, common words)

3. **PersonaPipeline**
   - Orchestrates the persona creation process
   - Handles data processing and result management
   - Provides detailed statistics and error tracking

## Usage

### Setting Up

1. Install required packages:
```bash
pip install openai pandas emoji python-dotenv
```

2. Set up your OpenAI API key:
```bash
export OPENAI_API_KEY='your-api-key'
```

### Using the Persona Pipeline

```python
from conversation_pipeline.components.persona.persona_pipeline import PersonaPipeline

# Initialize pipeline
pipeline = PersonaPipeline(
    api_key="your-openai-api-key",
    output_dir="path/to/output",
    cache_dir="path/to/cache",
    min_posts=10,
    max_posts=50
)

# Process user data
results = pipeline.process_user_data(
    data=your_dataframe,
    user_id_col='user_id',
    text_col='full_text',
    timestamp_col='created_at'
)

# Access results
personas = results['personas']
post_analyses = results['post_analyses']
stats = results['stats']

# Load previously generated results
loaded_results = PersonaPipeline.load_results("path/to/output")
```

### Input Data Format

The pipeline expects a pandas DataFrame with the following columns:
- `user_id`: Unique identifier for each user
- `full_text`: The text content of the post
- `created_at`: Timestamp of the post
- Additional columns are allowed but not required

### Output Structure

The pipeline generates three main outputs:

1. `personas.csv`: Contains generated user personas
   - User ID
   - Persona analysis
   - Generation metadata

2. `post_analyses.csv`: Contains detailed post analysis results
   - Post metrics
   - Temporal patterns
   - Content patterns
   - Interaction patterns
   - Language patterns

3. `pipeline_stats.txt`: Contains processing statistics
   - Total users processed
   - Success/error counts
   - Processing metrics

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
