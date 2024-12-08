import os
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Set
import sqlite3
import json
from dataclasses import dataclass, asdict
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ConversationMetadata:
    conversation_id: str
    start_time: datetime
    end_time: datetime
    participant_count: int
    tweet_count: int
    root_tweet_id: str
    root_user: str

@dataclass
class Tweet:
    tweet_id: str
    text: str
    author: str
    timestamp: datetime
    reply_to_id: Optional[str]
    reply_to_user: Optional[str]
    urls: List[str]
    
    @classmethod
    def from_pandas_row(cls, row):
        return cls(
            tweet_id=str(row['tweet_id']),
            text=row['full_text'],
            author=row['screen_name'],
            timestamp=pd.to_datetime(row['created_at']).tz_localize(None),
            reply_to_id=str(row['reply_to_id']) if pd.notna(row['reply_to_id']) else None,
            reply_to_user=row['reply_to_user'] if pd.notna(row['reply_to_user']) else None,
            urls=[url for url in [row['expandedURL']] if pd.notna(url)]
        )

class ConversationExtractor:
    def __init__(self):
        self.conversations: Dict[str, List[Tweet]] = {}
        self.processed_tweets: Set[str] = set()
        self.stats = {
            'processed_chunks': 0,
            'processed_tweets': 0,
            'conversations_found': 0,
            'orphaned_replies': 0
        }
    
    def process_chunk(self, df_chunk: pd.DataFrame) -> None:
        """Process a chunk of the dataset to extract conversations."""
        try:
            # Sort by timestamp to ensure chronological processing
            df_chunk = df_chunk.sort_values('created_at')
            
            for _, row in df_chunk.iterrows():
                tweet = Tweet.from_pandas_row(row)
                self.stats['processed_tweets'] += 1
                
                if tweet.tweet_id in self.processed_tweets:
                    continue
                    
                self.processed_tweets.add(tweet.tweet_id)
                
                # If it's a reply, add to existing conversation or start tracking a new one
                if tweet.reply_to_id:
                    if tweet.reply_to_id in self.conversations:
                        self.conversations[tweet.reply_to_id].append(tweet)
                    else:
                        # Start new conversation with this reply
                        self.conversations[tweet.reply_to_id] = [tweet]
                        self.stats['conversations_found'] += 1
                else:
                    # It's a root tweet, initialize new conversation
                    self.conversations[tweet.tweet_id] = [tweet]
            
            self.stats['processed_chunks'] += 1
            
            if self.stats['processed_chunks'] % 10 == 0:
                logger.info(f"Processed {self.stats['processed_chunks']} chunks, "
                          f"found {self.stats['conversations_found']} conversations, "
                          f"processed {self.stats['processed_tweets']} tweets")
                
        except Exception as e:
            logger.error(f"Error processing chunk: {str(e)}")
            raise

    def get_conversation_metadata(self, conversation_id: str, tweets: List[Tweet]) -> ConversationMetadata:
        """Generate metadata for a conversation."""
        if not tweets:
            raise ValueError(f"No tweets found for conversation {conversation_id}")
            
        timestamps = [tweet.timestamp for tweet in tweets]
        participants = {tweet.author for tweet in tweets}
        
        return ConversationMetadata(
            conversation_id=conversation_id,
            start_time=min(timestamps),
            end_time=max(timestamps),
            participant_count=len(participants),
            tweet_count=len(tweets),
            root_tweet_id=conversation_id,
            root_user=tweets[0].author
        )

    def validate_conversation(self, conversation_id: str, tweets: List[Tweet]) -> bool:
        """Validate a conversation for completeness and consistency."""
        if not tweets:
            return False
            
        # Check temporal consistency
        timestamps = [tweet.timestamp for tweet in tweets]
        if not all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1)):
            logger.warning(f"Temporal inconsistency in conversation {conversation_id}")
            return False
            
        # Check reply chain consistency
        reply_ids = {tweet.reply_to_id for tweet in tweets if tweet.reply_to_id}
        tweet_ids = {tweet.tweet_id for tweet in tweets}
        
        # All replies should be to tweets we know about
        if not reply_ids.issubset(tweet_ids):
            logger.warning(f"Incomplete reply chain in conversation {conversation_id}")
            return False
            
        return True

    def get_stats(self) -> Dict:
        """Return current processing statistics."""
        return self.stats
