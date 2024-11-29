import pandas as pd
import logging
from typing import Dict, List, Set

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
