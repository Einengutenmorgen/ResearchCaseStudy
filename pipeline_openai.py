# pipeline_openai.py

import pandas as pd
import asyncio
from aiohttp import ClientSession
from typing import Union, List, Dict
from openai import OpenAI
import pickle
from datetime import datetime
import os
from dotenv import load_dotenv
import tiktoken
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class DataProcessor:
    def __init__(self, csv_path: str):
        """Initialize the data processor with the path to the CSV file."""
        self.csv_path = csv_path
        self.df = None
        self.df_new = None
        
    def load_data(self):
        """Load and perform initial data processing."""
        self.df = pd.read_csv(self.csv_path)
        
        # Filter posts
        self.single_post_df = self.df[self.df[['reply_to_id', 'reply_to_user']].isna().all(axis=1)]
        self.reply_post_df = self.df[self.df[['reply_to_id', 'reply_to_user']].notna().any(axis=1)]
        
    def process_user_data(self):
        """Process and group data by user."""
        # Group by user
        user_groups = {}
        
        for _, row in self.df.iterrows():
            user_id = row['original_user_id']
            
            if user_id not in user_groups:
                user_groups[user_id] = {
                    'original_user_id': {user_id},
                    'full_text': [],
                    'tweet_id': [],
                    'created_at': [],
                    'screen_name': set(),
                    'retweeted_user_ID': [],
                    'collected_at': [],
                    'reply_to_id': [],
                    'reply_to_user': [],
                    'expandedURL': []
                }
            
            # Add data to group
            group = user_groups[user_id]
            group['full_text'].append(row['full_text'])
            group['tweet_id'].append(row['tweet_id'])
            group['created_at'].append(row['created_at'])
            group['screen_name'].add(row['screen_name'])
            group['retweeted_user_ID'].append(row['retweeted_user_ID'])
            group['collected_at'].append(row['collected_at'])
            group['reply_to_id'].append(row['reply_to_id'])
            group['reply_to_user'].append(row['reply_to_user'])
            group['expandedURL'].append(row['expandedURL'])
        
        # Convert to DataFrame
        self.df_new = pd.DataFrame([
            {**{'user_id': k}, **v} 
            for k, v in user_groups.items()
        ])
        
        return self.df_new

class PromptFormatter:
    """Formats prompts for OpenAI analysis."""
    
    @staticmethod
    def create_detailed_prompt(row: pd.Series, posts_for_persona: List[str]) -> str:
        """Create a detailed prompt for persona analysis."""
        # Format the tweets as a numbered list
        tweets_text = "\n".join([f"{i+1}. {tweet}" for i, tweet in enumerate(posts_for_persona)])
        
        prompt = f"""Analyze the following social media posts from a single user and create a detailed character description.
Focus on their personality traits, communication style, interests, and behavioral patterns.
Consider their humor style, emotional expressions, values, and social interaction patterns.

User's Posts:
{tweets_text}

Based on these posts, provide a comprehensive character description covering:
1. Humor Style
2. Communication Patterns
3. Emotional Expressions
4. Values and Beliefs
5. Interests and Hobbies
6. Social Interactions
7. Personality Traits
8. Cultural Background

Organize your analysis with clear sections and specific examples from the posts."""
        
        return prompt

class OpenAIAnalyzer:
    def __init__(self, api_key: str = None):
        """Initialize OpenAI client with API key."""
        self.client = OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
        self.encoding = tiktoken.encoding_for_model("gpt-4")
        
    def count_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Count the number of tokens in the messages."""
        num_tokens = 0
        for message in messages:
            # Count tokens in the content
            num_tokens += len(self.encoding.encode(message["content"]))
            # Add overhead for message format
            num_tokens += 4  # Every message follows <im_start>{role/name}\n{content}<im_end>\n
        num_tokens += 2  # Every reply is primed with <im_start>assistant
        return num_tokens
        
    def analyze_persona(self, prompt: str) -> str:
        """Analyze user persona using OpenAI."""
        try:
            messages = [
                {"role": "system", "content": "You are an expert at analyzing social media posts and creating detailed personality profiles."},
                {"role": "user", "content": prompt}
            ]
            
            # Count tokens before making the API call
            token_count = self.count_tokens(messages)
            logger.info(f"Total tokens in prompt: {token_count}")
            
            # GPT-4o has a 128k context window, check if we're within limits
            if token_count > 128000:
                logger.warning(f"Token count {token_count} exceeds model's context window of 128,000 tokens")
                return None
                
            completion = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.3,
                max_tokens=16384  # Maximum output tokens for gpt-4o
            )
            return completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in OpenAI API call: {e}")
            return None

class NeutralDescriptionGenerator:
    """Generates neutral descriptions of social media posts using OpenAI."""
    
    def __init__(self):
        self.openai_client = OpenAIAnalyzer()
    
    def create_neutral_prompt(self, post: str) -> str:
        """Create a prompt for neutral post description."""
        system_message = {
            "role": "system",
            "content": """You are a neutral observer tasked with creating objective, high-level descriptions of social media posts.
Focus on the type of content, general topic, and format.
Avoid subjective interpretations or unnecessary details.
Keep the description clear, concise, and unbiased."""
        }
        
        user_message = {
            "role": "user",
            "content": f"Create a neutral description of this social media post: {post}"
        }
        
        return [system_message, user_message]
    
    def generate_description(self, post: str) -> str:
        """Generate a neutral description for a single post."""
        prompt = self.create_neutral_prompt(post)
        try:
            response = self.openai_client.analyze_persona(prompt[1]["content"])
            return response.strip()
        except Exception as e:
            logging.error(f"Error generating neutral description: {e}")
            return "Error: Could not generate description"

def main(num_users: int = None):
    """
    Main function to process and analyze user data.
    
    Args:
        num_users (int, optional): Number of users to analyze. If None, analyzes all users.
    """
    # Initialize components
    csv_path = '/Users/mogen/Desktop/Research/storage/df_test_10k.csv'
    processor = DataProcessor(csv_path)
    formatter = PromptFormatter()
    analyzer = OpenAIAnalyzer()
    neutral_generator = NeutralDescriptionGenerator()
    
    # Process data
    processor.load_data()
    df_new = processor.process_user_data()
    
    # Limit number of users if specified
    if num_users is not None:
        df_new = df_new.head(num_users).copy()  # Create a copy to avoid SettingWithCopyWarning
        print(f"Analyzing personas for {num_users} users...")
    else:
        df_new = df_new.copy()  # Create a copy to avoid SettingWithCopyWarning
        print(f"Analyzing personas for all {len(df_new)} users...")
    
    # Create prompts and analyze
    personas = []
    neutral_descriptions = []
    skipped_users = []
    
    for idx, row in df_new.iterrows():
        print(f"Processing user {idx + 1}/{len(df_new)}...")
        
        # Get all tweets for the user
        tweets = eval(row['full_text']) if isinstance(row['full_text'], str) else row['full_text']
        
        # Check if user has enough posts
        if len(tweets) < 70:  # Need at least 50 for persona + 20 for neutral descriptions
            print(f"Skipping user {row['user_id']} - insufficient posts ({len(tweets)} posts)")
            skipped_users.append(row['user_id'])
            continue
        
        # Split posts: first 50 for persona, next 20 for neutral descriptions
        posts_for_persona = tweets[:50]
        posts_for_neutral = tweets[50:70]
        
        # Generate persona using first 50 posts
        prompt = formatter.create_detailed_prompt(row, posts_for_persona)
        persona = analyzer.analyze_persona(prompt)
        personas.append(persona)
        
        # Generate neutral descriptions for next 20 posts
        descriptions = []
        print("Generating neutral descriptions for tweets...")
        for tweet in posts_for_neutral:
            description = neutral_generator.generate_description(tweet)
            descriptions.append(description)
        
        neutral_descriptions.append(descriptions)
        
        # Print the analysis for verification
        print("\nPersona Analysis (based on first 50 posts):")
        print(persona)
        print("\nSample Neutral Descriptions (from next 20 posts):")
        for i, (tweet, desc) in enumerate(zip(posts_for_neutral[:3], descriptions[:3]), 1):
            print(f"\nTweet {i}:")
            print(f"Original: {tweet}")
            print(f"Neutral Description: {desc}")
        print("-" * 80)
    
    # Remove skipped users from dataframe
    df_new = df_new[~df_new['user_id'].isin(skipped_users)].copy()
    
    # Add results to dataframe
    df_new['persona'] = personas
    df_new['neutral_descriptions'] = neutral_descriptions
    df_new['posts_used_for_persona'] = [50] * len(personas)
    df_new['posts_used_for_neutral'] = [20] * len(personas)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f'analyzed_personas_{timestamp}.csv'
    
    # Convert sets to strings in the DataFrame to ensure proper CSV saving
    for col in df_new.columns:
        if df_new[col].apply(lambda x: isinstance(x, set)).any():
            df_new[col] = df_new[col].apply(lambda x: str(x) if isinstance(x, set) else x)
    
    df_new.to_csv(output_path, index=False, quoting=1)  # Use quoting=1 to properly handle text fields
    
    # Print summary
    print(f"\nAnalysis Summary:")
    print(f"Total users processed: {len(df_new)}")
    print(f"Users skipped (insufficient posts): {len(skipped_users)}")
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze user personas from social media data')
    parser.add_argument('--num_users', type=int, help='Number of users to analyze (default: all users)', default=None)
    
    args = parser.parse_args()
    main(args.num_users)