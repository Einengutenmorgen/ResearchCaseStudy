import json
import logging
from pathlib import Path
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RootTweetFetcher:
    def __init__(self, data_file: str):
        """
        Initialize the RootTweetFetcher.
        
        Args:
            data_file: Path to the CSV file containing all tweets
        """
        self.data_file = data_file
        self.root_tweets_cache = {}
        
    def process_conversation_batch(self, conversation_dir: str, output_dir: Optional[str] = None) -> Dict[str, int]:
        """
        Process a batch of conversations and add root tweets.
        
        Args:
            conversation_dir: Directory containing conversation JSON files
            output_dir: Optional output directory for enhanced conversations
            
        Returns:
            Statistics about the processing
        """
        try:
            conv_dir = Path(conversation_dir)
            if not conv_dir.exists():
                raise ValueError(f"Conversation directory {conversation_dir} does not exist")
                
            output_dir = Path(output_dir) if output_dir else conv_dir / "with_roots"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Get all root tweet IDs from conversations
            root_ids = self._collect_root_ids(conv_dir)
            logger.info(f"Found {len(root_ids)} unique root tweet IDs")
            
            # Fetch root tweets
            root_tweets = self._fetch_root_tweets(root_ids)
            logger.info(f"Fetched {len(root_tweets)} root tweets")
            
            # Enhance conversations with root tweets
            stats = self._enhance_conversations(conv_dir, output_dir, root_tweets)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error processing conversation batch: {str(e)}")
            raise
            
    def _collect_root_ids(self, conv_dir: Path) -> set:
        """
        Collect all unique root tweet IDs from conversations.
        """
        root_ids = set()
        for conv_file in conv_dir.glob("conversation_*.json"):
            try:
                with open(conv_file) as f:
                    conv = json.load(f)
                    root_ids.add(conv["root_id"])
            except Exception as e:
                logger.warning(f"Error reading conversation file {conv_file}: {str(e)}")
                
        return root_ids
        
    def _fetch_root_tweets(self, root_ids: set) -> Dict[str, Dict]:
        """
        Fetch root tweets from the data file.
        """
        try:
            # Convert root_ids to numeric format
            numeric_root_ids = {float(id_str) for id_str in root_ids}
            logger.info(f"Looking for root tweets with IDs: {numeric_root_ids}")
            
            # Read data in chunks to handle large files
            chunk_size = 10000
            root_tweets = {}
            
            for chunk in pd.read_csv(self.data_file, chunksize=chunk_size):
                # Filter for root tweets
                matches = chunk[chunk['tweet_id'].isin(numeric_root_ids)]
                
                for _, tweet in matches.iterrows():
                    tweet_id = str(int(tweet['tweet_id']))  # Convert to string for consistent keys
                    root_tweets[tweet_id] = tweet.to_dict()
                    logger.info(f"Found root tweet: {tweet_id}")
                    
                # Break if we found all root tweets
                if len(root_tweets) == len(root_ids):
                    break
                    
            return root_tweets
            
        except Exception as e:
            logger.error(f"Error fetching root tweets: {str(e)}")
            raise
            
    def _enhance_conversations(
        self, 
        conv_dir: Path, 
        output_dir: Path, 
        root_tweets: Dict[str, Dict]
    ) -> Dict[str, int]:
        """
        Enhance conversations with their root tweets.
        """
        stats = {
            "total_conversations": 0,
            "conversations_with_root": 0,
            "conversations_missing_root": 0
        }
        
        try:
            for conv_file in conv_dir.glob("conversation_*.json"):
                stats["total_conversations"] += 1
                
                # Read conversation
                with open(conv_file) as f:
                    conv = json.load(f)
                    
                # Add root tweet if available
                root_id = conv["root_id"]
                if root_id in root_tweets:
                    root_tweet = root_tweets[root_id]
                    conv["root_tweet"] = root_tweet
                    conv["messages"].insert(0, root_tweet)  # Add as first message
                    conv["participants"].append(str(root_tweet["original_user_id"]))
                    conv["metadata"]["length"] += 1
                    
                    # Update time span if needed
                    root_time = pd.to_datetime(root_tweet["created_at"])
                    start_time = pd.to_datetime(conv["metadata"]["time_span"]["start"])
                    if root_time < start_time:
                        conv["metadata"]["time_span"]["start"] = str(root_time)
                        conv["metadata"]["time_span"]["duration_seconds"] = (
                            pd.to_datetime(conv["metadata"]["time_span"]["end"]) - root_time
                        ).total_seconds()
                        
                    stats["conversations_with_root"] += 1
                else:
                    stats["conversations_missing_root"] += 1
                    
                # Save enhanced conversation
                output_file = output_dir / conv_file.name
                with open(output_file, 'w') as f:
                    json.dump(conv, f, indent=2, default=str)
                    
            logger.info(f"Enhanced conversation statistics: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Error enhancing conversations: {str(e)}")
            raise
