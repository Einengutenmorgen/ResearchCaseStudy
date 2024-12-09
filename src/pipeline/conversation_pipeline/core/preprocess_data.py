import pandas as pd
import os
import re
import emoji
from typing import Tuple
from datetime import datetime
import logging
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, input_file: str):
        """
        Initialize the preprocessor with input file path.
        
        Args:
            input_file: Path to the input CSV file
        """
        self.input_file = input_file
        self.df = None
        self.posts_df = None
        self.replies_df = None
        
        # Create results directory if it doesn't exist
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join("Results", f"processed_data_{timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_data(self) -> None:
        """Load the input CSV file."""
        logger.info(f"Loading data from {self.input_file}")
        try:
            self.df = pd.read_csv(self.input_file)
            logger.info(f"Loaded {len(self.df)} rows")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
            
    def split_posts_replies(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into posts and replies based on reply_to_id and reply_to_user.
        
        Returns:
            Tuple of (posts_df, replies_df)
        """
        logger.info("Splitting data into posts and replies")
        
        # Identify replies (posts that have either reply_to_id or reply_to_user)
        is_reply = self.df[['reply_to_id', 'reply_to_user']].notna().any(axis=1)
        
        self.posts_df = self.df[~is_reply].copy()
        self.replies_df = self.df[is_reply].copy()
        
        logger.info(f"Split into {len(self.posts_df)} posts and {len(self.replies_df)} replies")
        return self.posts_df, self.replies_df
    
    @staticmethod
    def count_words(text: str) -> int:
        """Count the number of words in text, excluding mentions and URLs."""
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        # Remove mentions
        text = re.sub(r'@\w+', '', text)
        # Split and count non-empty words
        words = [word for word in text.split() if word.strip()]
        return len(words)
    
    @staticmethod
    def count_mentions(text: str) -> int:
        """Count the number of mentions (@) in text."""
        return len(re.findall(r'@\w+', text))
    
    @staticmethod
    def is_emoji_only(text: str) -> bool:
        """Check if text consists only of emojis and whitespace."""
        # Remove all emojis
        text_without_emoji = ''.join(char for char in text if char not in emoji.EMOJI_DATA)
        # Check if remaining text is empty or only whitespace
        return not text_without_emoji.strip()
    
    def filter_posts(self) -> pd.DataFrame:
        """
        Filter posts based on criteria:
        - Remove entries with less than 5 words
        - Remove entries that only consist of emojis
        - Remove entries with more than one mention
        
        Returns:
            Filtered posts DataFrame
        """
        if self.posts_df is None:
            raise ValueError("Posts DataFrame not initialized. Run split_posts_replies first.")
            
        logger.info("Filtering posts based on criteria")
        initial_count = len(self.posts_df)
        
        # Add filter columns
        self.posts_df['word_count'] = self.posts_df['full_text'].apply(self.count_words)
        self.posts_df['mention_count'] = self.posts_df['full_text'].apply(self.count_mentions)
        self.posts_df['is_emoji_only'] = self.posts_df['full_text'].apply(self.is_emoji_only)
        
        # Apply filters
        filtered_posts = self.posts_df[
            (self.posts_df['word_count'] >= 5) &
            (~self.posts_df['is_emoji_only']) &
            (self.posts_df['mention_count'] <= 1)
        ].copy()
        
        # Remove helper columns
        filtered_posts.drop(['word_count', 'mention_count', 'is_emoji_only'], axis=1, inplace=True)
        
        removed_count = initial_count - len(filtered_posts)
        logger.info(f"Removed {removed_count} posts ({removed_count/initial_count*100:.1f}%)")
        
        return filtered_posts
    
    def save_processed_data(self) -> Tuple[str, str]:
        """
        Save processed posts and replies to CSV files.
        
        Returns:
            Tuple of (posts_file_path, replies_file_path)
        """
        # Save posts
        posts_file = os.path.join(self.output_dir, "filtered_posts.csv")
        self.posts_df.to_csv(posts_file, index=False)
        logger.info(f"Saved filtered posts to {posts_file}")
        
        # Save replies
        replies_file = os.path.join(self.output_dir, "replies.csv")
        self.replies_df.to_csv(replies_file, index=False)
        logger.info(f"Saved replies to {replies_file}")
        
        return posts_file, replies_file
    
    def process(self) -> Tuple[str, str]:
        """
        Run the complete preprocessing pipeline.
        
        Returns:
            Tuple of (posts_file_path, replies_file_path)
        """
        self.load_data()
        self.split_posts_replies()
        self.posts_df = self.filter_posts()
        return self.save_processed_data()

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Preprocess tweet data by splitting and filtering posts.')
    parser.add_argument('--input', '-i', 
                      type=str,
                      default='data/tweets.csv',
                      help='Path to input CSV file (default: data/tweets.csv)')
    
    args = parser.parse_args()
    
    try:
        preprocessor = DataPreprocessor(args.input)
        posts_file, replies_file = preprocessor.process()
        
        logger.info("Preprocessing completed successfully!")
        logger.info(f"Filtered posts saved to: {posts_file}")
        logger.info(f"Replies saved to: {replies_file}")
        
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise

if __name__ == "__main__":
    main()
