import pandas as pd
import logging
from typing import Dict, List, Set
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UserPostCollector:
    def __init__(self, data_file: str, max_posts: int = 50):
        """
        Initialize the UserPostCollector.
        
        Args:
            data_file: Path to the CSV file containing all tweets
            max_posts: Maximum number of posts to collect per user for persona generation
        """
        self.data_file = data_file
        self.max_posts = max_posts
        
    def collect_user_posts(self, user_ids: Set) -> Dict[str, List[Dict]]:
        """
        Collect non-conversation posts for specified users.
        
        Args:
            user_ids: Set of user IDs to collect posts for
            
        Returns:
            Dictionary mapping user IDs to their posts
        """
        logger.info(f"Collecting posts for {len(user_ids)} users")
        user_posts = {str(uid): [] for uid in user_ids}
        
        try:
            # Read data in chunks to handle large files
            for chunk in pd.read_csv(self.data_file, chunksize=10000):
                # Convert user IDs to string for consistent comparison
                chunk['original_user_id'] = chunk['original_user_id'].astype(str)
                
                # Filter for posts by target users
                user_chunk = chunk[chunk['original_user_id'].isin(user_ids)]
                
                # Group by user and collect posts
                for user_id in user_ids:
                    user_id_str = str(user_id)
                    user_data = user_chunk[user_chunk['original_user_id'] == user_id_str]
                    
                    # Add posts to user's collection
                    current_posts = user_posts[user_id_str]
                    if len(current_posts) < self.max_posts:
                        new_posts = user_data.to_dict('records')
                        current_posts.extend(new_posts)
                        user_posts[user_id_str] = current_posts[:self.max_posts]
                
                # Check if we have enough posts for all users
                if all(len(posts) >= self.max_posts for posts in user_posts.values()):
                    break
                    
            # Log collection results
            for user_id, posts in user_posts.items():
                logger.info(f"Collected {len(posts)} posts for user {user_id}")
                
            return user_posts
            
        except Exception as e:
            logger.error(f"Error collecting user posts: {str(e)}")
            raise
