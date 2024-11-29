import os
import pandas as pd
import logging
import sys
sys.path.append('/Users/mogen/Desktop/Research/src/pipeline')
from main_pipeline import Pipeline
from tabulate import tabulate
from typing import Dict, List
from similarity_analyzer import SimilarityAnalyzer, SimilarityMetric

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_filter_data(input_file: str, num_users: int = 2) -> pd.DataFrame:
    """
    Load the input data and filter for specified number of users with sufficient posts.
    
    Args:
        input_file: Path to input CSV file
        num_users: Number of users to process
    
    Returns:
        DataFrame with filtered data
    """
    df = pd.read_csv(input_file)
    
    # Group by user and count posts
    user_posts = df.groupby('original_user_id').size().reset_index(name='post_count')
    # Filter users with at least 70 posts (50 for persona + 20 for testing)
    eligible_users = user_posts[user_posts['post_count'] >= 70].head(num_users)
    
    # Filter original dataframe for selected users
    filtered_df = df[df['original_user_id'].isin(eligible_users['original_user_id'])]
    return filtered_df

def analyze_results(results_dir: str) -> None:
    """
    Analyze and display the results of the regeneration pipeline.
    
    Args:
        results_dir: Directory containing the results files
    """
    # Load the results
    regeneration_file = max(
        [f for f in os.listdir(results_dir) if f.startswith('regenerated_posts_')],
        key=lambda x: os.path.getctime(os.path.join(results_dir, x))
    )
    
    results_df = pd.read_csv(os.path.join(results_dir, regeneration_file))
    
    # Initialize similarity analyzer
    analyzer = SimilarityAnalyzer()
    metrics = [
        SimilarityMetric.SEMANTIC,
        SimilarityMetric.COSINE,
        SimilarityMetric.LLM,
        SimilarityMetric.COMBINED
    ]
    
    # Group by user
    for user_id in results_df['user_id'].unique():
        user_results = results_df[results_df['user_id'] == user_id]
        
        print(f"\n{'='*80}")
        print(f"Analysis for User {user_id}")
        print(f"{'='*80}")
        
        # Display persona
        print("\nPersona Analysis:")
        print("-" * 40)
        print(user_results['persona'].iloc[0])
        
        print("\nPost Comparisons and Similarity Analysis:")
        print("-" * 40)
        
        # Prepare posts for batch analysis
        posts_for_analysis = []
        for _, row in user_results.iterrows():
            posts_for_analysis.append({
                'original_post': row['original_post'],
                'regenerated_post': row['regenerated_post'],
                'neutral_description': row['neutral_description']
            })
        
        # Run similarity analysis
        similarity_results = analyzer.analyze_batch(posts_for_analysis, metrics)
        
        # Display results with similarity scores
        for idx, (result, row) in enumerate(zip(similarity_results, user_results.iterrows())):
            print(f"\nPost {idx + 1}:")
            print("\nOriginal Post:")
            print(f"  {result['original_post']}")
            print("\nNeutral Description:")
            print(f"  {result['neutral_description']}")
            print("\nRegenerated Post:")
            print(f"  {result['regenerated_post']}")
            print("\nSimilarity Scores:")
            print(f"  Semantic Similarity: {result['semantic_similarity']:.3f}")
            print(f"  Cosine Similarity: {result['cosine_similarity']:.3f}")
            if 'llm_similarity' in result:
                print(f"  LLM Similarity: {result['llm_similarity']:.3f}")
                print(f"  LLM Explanation: {result['llm_explanation']}")
            print(f"  Combined Score: {result['combined_score']:.3f}")
            print("\n" + "-" * 40)
        
        # Calculate and display average scores for the user
        avg_scores = {
            'semantic': sum(r['semantic_similarity'] for r in similarity_results) / len(similarity_results),
            'cosine': sum(r['cosine_similarity'] for r in similarity_results) / len(similarity_results),
            'combined': sum(r['combined_score'] for r in similarity_results) / len(similarity_results)
        }
        
        if 'llm_similarity' in similarity_results[0]:
            avg_scores['llm'] = sum(r['llm_similarity'] for r in similarity_results) / len(similarity_results)
        
        print("\nAverage Similarity Scores:")
        print(f"  Semantic Similarity: {avg_scores['semantic']:.3f}")
        print(f"  Cosine Similarity: {avg_scores['cosine']:.3f}")
        if 'llm' in avg_scores:
            print(f"  LLM Similarity: {avg_scores['llm']:.3f}")
        print(f"  Combined Score: {avg_scores['combined']:.3f}")
        print("\n")

def main():
    # Set up paths
    input_file = '/Users/mogen/Desktop/Research/data/df_test_10k.csv'  # Updated path to correct input file
    output_dir = os.path.join("Results", f"test_regeneration")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Filter data for 2 users
        filtered_df = load_and_filter_data(input_file, num_users=2)
        filtered_file = os.path.join(output_dir, "filtered_input.csv")
        filtered_df.to_csv(filtered_file, index=False)
        
        # Run pipeline
        logger.info("Starting pipeline with filtered dataset")
        pipeline = Pipeline(filtered_file, output_dir)
        results = pipeline.run()
        
        # Analyze results
        logger.info("Analyzing results")
        analyze_results(output_dir)
        
        logger.info("Test completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during test: {e}")
        raise

if __name__ == "__main__":
    main()
