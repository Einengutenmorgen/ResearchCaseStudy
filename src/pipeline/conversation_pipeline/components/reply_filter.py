import pandas as pd
import logging
from typing import Optional, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReplyFilter:
    def __init__(self):
        """Initialize the ReplyFilter."""
        self.required_columns = ['tweet_id', 'original_user_id', 'reply_to_id', 'created_at', 'full_text']
        
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate that the data contains required columns.
        
        Args:
            data: Input DataFrame
            
        Returns:
            True if valid, raises Exception if not
        """
        missing_cols = [col for col in self.required_columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        return True
        
    def filter_replies(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Filter for reply posts and clean the data.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame containing only valid replies
        """
        try:
            self.validate_data(data)
            
            # Filter for posts that are replies
            replies = data[data['reply_to_id'].notna()].copy()
            
            # Clean and standardize
            replies['created_at'] = pd.to_datetime(replies['created_at'])
            replies['reply_to_id'] = replies['reply_to_id'].astype('Int64')
            
            # Sort by creation time
            replies.sort_values('created_at', inplace=True)
            
            logger.info(f"Filtered {len(replies)} replies from {len(data)} total posts")
            return replies
            
        except Exception as e:
            logger.error(f"Error filtering replies: {str(e)}")
            raise
            
    def extract_metadata(self, replies: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract metadata about the replies.
        
        Args:
            replies: DataFrame of filtered replies
            
        Returns:
            Dictionary containing metadata
        """
        try:
            metadata = {
                'total_replies': len(replies),
                'unique_users': replies['original_user_id'].nunique(),
                'date_range': {
                    'start': replies['created_at'].min(),
                    'end': replies['created_at'].max()
                },
                'reply_chains': self._analyze_reply_chains(replies)
            }
            
            logger.info(f"Extracted metadata: {metadata}")
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting metadata: {str(e)}")
            raise
            
    def _analyze_reply_chains(self, replies: pd.DataFrame) -> Dict[str, int]:
        """
        Analyze reply chain characteristics.
        
        Args:
            replies: DataFrame of filtered replies
            
        Returns:
            Dictionary with reply chain statistics
        """
        # Count how many times each status is replied to
        reply_counts = replies['reply_to_id'].value_counts()
        
        return {
            'total_chains': len(reply_counts),
            'max_replies_to_single_post': reply_counts.max(),
            'avg_replies_per_post': reply_counts.mean()
        }
