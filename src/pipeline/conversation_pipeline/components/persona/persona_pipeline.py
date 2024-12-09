"""Pipeline for creating and analyzing user personas."""

import logging
import os
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
from pathlib import Path

from src.pipeline.conversation_pipeline.components.persona.persona_creator import PersonaCreator
from src.pipeline.conversation_pipeline.components.persona.post_analyzer import PostAnalyzer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PersonaPipeline:
    """Pipeline for orchestrating the persona creation and analysis process."""
    
    def __init__(self, 
                 api_key: str,
                 output_dir: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 min_posts: int = 40,
                 max_posts: int = 50):
        """
        Initialize the persona pipeline.
        
        Args:
            api_key: OpenAI API key
            output_dir: Directory to save results (default: creates timestamped directory)
            cache_dir: Directory to cache personas
            min_posts: Minimum number of posts required for analysis
            max_posts: Maximum number of posts to use for analysis
        """
        self.min_posts = min_posts
        self.max_posts = max_posts
        
        # Set up output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = Path("Results") / f"persona_analysis_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.persona_creator = PersonaCreator(api_key, cache_dir)
        self.post_analyzer = PostAnalyzer()
        
        logger.info(f"Initialized PersonaPipeline with output directory: {self.output_dir}")

    def process_user_data(self, 
                         data: pd.DataFrame,
                         user_id_col: str = 'user_id',
                         text_col: str = 'full_text',
                         timestamp_col: str = 'created_at') -> Dict:
        """
        Process user data to create personas and analyze posting patterns.
        
        Args:
            data: DataFrame containing user posts
            user_id_col: Name of user ID column
            text_col: Name of text content column
            timestamp_col: Name of timestamp column
            
        Returns:
            Dictionary containing processing results and statistics
        """
        try:
            logger.info("Starting user data processing")
            
            # Group posts by user
            user_groups = data.groupby(user_id_col)
            total_users = len(user_groups)
            
            results = {
                "personas": [],
                "post_analyses": [],
                "stats": {
                    "total_users": total_users,
                    "processed_users": 0,
                    "skipped_users": 0,
                    "errors": 0
                }
            }
            
            for user_id, user_data in user_groups:
                try:
                    # Convert user posts to list of dictionaries
                    posts = user_data.to_dict('records')
                    
                    # Skip users with too few posts
                    if len(posts) < self.min_posts:
                        logger.debug(f"Skipping user {user_id}: insufficient posts ({len(posts)} < {self.min_posts})")
                        results["stats"]["skipped_users"] += 1
                        continue
                    
                    # Limit posts for analysis
                    analysis_posts = posts[:self.max_posts]
                    
                    # Analyze posts
                    post_analysis = self.post_analyzer.analyze_posts(analysis_posts)
                    results["post_analyses"].append({
                        "user_id": user_id,
                        "analysis": post_analysis
                    })
                    
                    # Generate persona
                    persona = self.persona_creator.create_persona(user_id, analysis_posts)
                    results["personas"].append(persona)
                    
                    results["stats"]["processed_users"] += 1
                    logger.info(f"Processed user {user_id} ({results['stats']['processed_users']}/{total_users})")
                    
                except Exception as e:
                    logger.error(f"Error processing user {user_id}: {str(e)}")
                    results["stats"]["errors"] += 1
            
            # Save results
            self._save_results(results)
            
            logger.info("User data processing completed")
            return results
            
        except Exception as e:
            logger.error(f"Error in pipeline execution: {str(e)}")
            raise

    def _save_results(self, results: Dict) -> None:
        """
        Save pipeline results to output directory.
        
        Args:
            results: Dictionary containing pipeline results
        """
        try:
            # Save personas
            personas_df = pd.DataFrame(results["personas"])
            personas_file = self.output_dir / "personas.csv"
            personas_df.to_csv(personas_file, index=False)
            
            # Save post analyses
            analyses_df = pd.DataFrame(results["post_analyses"])
            analyses_file = self.output_dir / "post_analyses.csv"
            analyses_df.to_csv(analyses_file, index=False)
            
            # Save statistics
            stats_file = self.output_dir / "pipeline_stats.txt"
            with open(stats_file, 'w') as f:
                f.write("Pipeline Statistics:\n")
                for key, value in results["stats"].items():
                    f.write(f"{key}: {value}\n")
            
            logger.info(f"Results saved to {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise

    @staticmethod
    def load_results(output_dir: str) -> Dict:
        """
        Load previously generated results from an output directory.
        
        Args:
            output_dir: Directory containing pipeline results
            
        Returns:
            Dictionary containing loaded results
        """
        try:
            output_dir = Path(output_dir)
            
            results = {
                "personas": pd.read_csv(output_dir / "personas.csv").to_dict('records'),
                "post_analyses": pd.read_csv(output_dir / "post_analyses.csv").to_dict('records')
            }
            
            # Load statistics
            stats_file = output_dir / "pipeline_stats.txt"
            stats = {}
            with open(stats_file) as f:
                for line in f.readlines()[1:]:  # Skip header
                    key, value = line.strip().split(': ')
                    stats[key] = int(value)
            
            results["stats"] = stats
            return results
            
        except Exception as e:
            logger.error(f"Error loading results from {output_dir}: {str(e)}")
            raise
