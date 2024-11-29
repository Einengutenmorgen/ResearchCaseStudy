import os
import logging
import argparse
from datetime import datetime
from typing import Optional
import pandas as pd
import time

from preprocess_data import DataPreprocessor
from data_processor import DataProcessor
from prompt_formatter import PromptFormatter
from openai_analyzer import OpenAIAnalyzer
from enhanced_neutral_generator import EnhancedNeutralDescriptionGenerator, GeneratorConfig, DescriptionStyle, OutputFormat
from post_regenerator import PostRegenerator, RegenerationResultsManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Pipeline:
    def __init__(self, input_file: str, output_dir: Optional[str] = None):
        """
        Initialize the pipeline with input file and output directory.
        
        Args:
            input_file: Path to input CSV file
            output_dir: Optional output directory path
        """
        self.input_file = input_file
        self.output_dir = output_dir or os.path.join("Results", f"pipeline_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize components
        self.preprocessor = DataPreprocessor(input_file)
        self.data_processor = DataProcessor(input_file)
        self.openai_analyzer = OpenAIAnalyzer()
        
        # Configure enhanced neutral description generator
        config = GeneratorConfig(
            model_name="gpt-4o",
            description_style=DescriptionStyle.DETAILED,
            output_format=OutputFormat.STRUCTURED,
            include_metadata=True,
            focus_aspects=["tone", "topic", "intent", "style"],
            min_post_length=5,
            max_post_length=1000,
            batch_size=5,
            timeout=45,
            max_tokens=8192
        )
        self.neutral_generator = EnhancedNeutralDescriptionGenerator(config)
        
    def run(self):
        """Execute the complete pipeline."""
        try:
            # Step 1: Preprocess data
            logger.info("Step 1: Preprocessing data")
            posts_file, replies_file = self.preprocessor.process()
            
            # Step 2: Process user data
            logger.info("Step 2: Processing user data")
            self.data_processor.load_data()
            processed_data = self.data_processor.process_user_data()
            
            # Step 3: Generate persona analyses using first 50 posts
            logger.info("Step 3: Generating persona analysis from first 50 posts")
            persona_analyses = []
            total_users = len(processed_data)
            
            for idx, (_, row) in enumerate(processed_data.iterrows()):
                logger.info(f"Processing user {idx + 1}/{total_users}")
                posts = row['full_text']
                if isinstance(posts, list) and len(posts) >= 50:
                    # Take first 50 posts for persona analysis
                    persona_posts = posts[:50]
                    prompt = PromptFormatter.create_detailed_prompt(row, persona_posts)
                    analysis = self.openai_analyzer.analyze_persona(prompt)
                    if analysis:
                        persona_analyses.append({
                            'user_id': row['user_id'],
                            'posts_used': len(persona_posts),
                            'persona_analysis': analysis
                        })
                        logger.info(f"Generated persona analysis for user {row['user_id']}")
                    
                    # Add delay every 3 requests to avoid rate limiting
                    if idx % 99 == 98:
                        logger.info("Rate limit pause - waiting 60 seconds...")
                        time.sleep(60)  # 60-second delay
            
            # Save persona analyses
            analyses_df = pd.DataFrame(persona_analyses)
            analyses_file = os.path.join(self.output_dir, "persona_analyses.csv")
            analyses_df.to_csv(analyses_file, index=False)
            logger.info(f"Saved {len(persona_analyses)} persona analyses to {analyses_file}")
            
            # Step 4: Generate neutral descriptions for posts 51-70
            logger.info("Step 4: Generating neutral descriptions for posts 51-70")
            descriptions = []
            processed_count = 0
            
            for idx, (_, row) in enumerate(processed_data.iterrows()):
                logger.info(f"Processing user {idx + 1}/{total_users}")
                posts = row['full_text']
                if isinstance(posts, list) and len(posts) > 50:
                    # Take posts 51-70 for neutral descriptions
                    description_posts = posts[50:70]
                    for i, post in enumerate(description_posts, start=51):
                        description = self.neutral_generator.generate_description(post)
                        if description:
                            descriptions.append({
                                'user_id': row['user_id'],
                                'post_number': i,
                                'original_post': post,
                                'neutral_description': description
                            })
                            processed_count += 1
                            if processed_count % 10 == 0:
                                logger.info(f"Processed {processed_count} descriptions")
                        
                        # Add delay every 3 requests to avoid rate limiting
                        if (idx * len(description_posts) + i) % 99 == 98:
                            logger.info("Rate limit pause - waiting 60 seconds...")
                            time.sleep(60)  # 60-second delay
            
            # Save neutral descriptions
            descriptions_df = pd.DataFrame(descriptions)
            descriptions_file = os.path.join(self.output_dir, "neutral_descriptions.csv")
            descriptions_df.to_csv(descriptions_file, index=False)
            logger.info(f"Saved {len(descriptions)} neutral descriptions to {descriptions_file}")
            
            # Step 5: Generate new posts based on personas and neutral descriptions
            logger.info("Step 5: Generating new posts based on personas")
            post_regenerator = PostRegenerator()
            results_manager = RegenerationResultsManager(self.output_dir)
            regeneration_results = []
            
            for _, persona_row in analyses_df.iterrows():
                user_descriptions = descriptions_df[descriptions_df['user_id'] == persona_row['user_id']]
                
                for _, desc_row in user_descriptions.iterrows():
                    regenerated_post = post_regenerator.regenerate_post(
                        persona_row['persona_analysis'],
                        desc_row['neutral_description']
                    )
                    
                    if regenerated_post:
                        regeneration_results.append({
                            'user_id': persona_row['user_id'],
                            'original_post': desc_row['original_post'],
                            'neutral_description': desc_row['neutral_description'],
                            'regenerated_post': regenerated_pos