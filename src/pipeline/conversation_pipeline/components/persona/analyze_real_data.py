"""Script for analyzing real data using the persona creation pipeline."""

import os
import pandas as pd
import logging
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path

from src.pipeline.conversation_pipeline.components.persona.persona_creator import PersonaCreator
from src.pipeline.conversation_pipeline.components.persona.post_analyzer import PostAnalyzer
from src.pipeline.conversation_pipeline.components.persona.persona_pipeline import PersonaPipeline

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_and_prepare_data(file_path: str) -> pd.DataFrame:
    """
    Load and prepare data for analysis.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Prepared DataFrame
    """
    logger.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    
    # Map columns to expected names
    df['user_id'] = df['original_user_id']
    df['created_at'] = pd.to_datetime(df['created_at'])
    
    # Ensure required columns exist
    required_columns = ['user_id', 'full_text', 'created_at']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Sort by user_id and created_at
    df = df.sort_values(['user_id', 'created_at'])
    
    logger.info(f"Loaded {len(df)} posts from {df['user_id'].nunique()} users")
    return df

def analyze_sample(df: pd.DataFrame, sample_size: int = 30) -> None:
    """
    Analyze a sample of users from the dataset.
    
    Args:
        df: Input DataFrame
        sample_size: Number of users to analyze
    """
    # Get users with sufficient posts
    user_post_counts = df['user_id'].value_counts()
    eligible_users = user_post_counts[user_post_counts >= 40].index
    
    if len(eligible_users) < sample_size:
        logger.warning(f"Only {len(eligible_users)} users have sufficient posts")
        sample_size = len(eligible_users)
    
    # Sample users
    sample_users = pd.Series(eligible_users).sample(sample_size)
    
    # Initialize pipeline
    pipeline = PersonaPipeline(
        api_key=os.getenv('OPENAI_API_KEY'),
        output_dir=f"results/persona_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        cache_dir="cache/personas",
        min_posts=40,
        max_posts=50
    )
    
    # Process sample
    sample_df = df[df['user_id'].isin(sample_users)]
    results = pipeline.process_user_data(sample_df)
    
    # Print summary
    logger.info("\nAnalysis Results Summary:")
    logger.info(f"Total users processed: {results['stats']['processed_users']}")
    logger.info(f"Successful analyses: {results['stats']['processed_users'] - results['stats']['errors']}")
    logger.info(f"Errors: {results['stats']['errors']}")
    logger.info(f"\nResults saved to: {pipeline.output_dir}")
    
    # Print sample of results
    logger.info("\nSample Persona Analysis:")
    for i, persona in enumerate(results['personas'][:3], 1):  # Show first 3 personas
        logger.info(f"\nUser {i}: {persona['user_id']}")
        logger.info("Analysis:")
        logger.info(persona['analysis'])
        logger.info("-" * 80)

def main():
    """Run the analysis on real data."""
    load_dotenv()
    
    # Check OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        raise ValueError("OpenAI API key not found in environment variables")
    
    try:
        # Load and analyze data
        df = load_and_prepare_data('/Users/mogen/Desktop/Research/data/df_test_10k.csv')
        analyze_sample(df, sample_size=30)
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()
