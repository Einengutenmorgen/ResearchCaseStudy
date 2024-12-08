import os
import json
import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import asdict

from conversation_pipeline import Tweet, ConversationMetadata

logger = logging.getLogger(__name__)

class ConversationStorageManager:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.json_dir = self.output_dir / "conversation_files"
        self.db_path = self.output_dir / "conversations.db"
        
        # Create directories
        self.json_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        self.stats = {
            'conversations_saved': 0,
            'tweets_saved': 0,
            'storage_errors': 0
        }

    def _init_database(self):
        """Initialize SQLite database with required schema."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create conversations table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS conversations (
                        conversation_id TEXT PRIMARY KEY,
                        start_time TIMESTAMP,
                        end_time TIMESTAMP,
                        participant_count INTEGER,
                        tweet_count INTEGER,
                        root_tweet_id TEXT,
                        root_user TEXT
                    )
                ''')
                
                # Create tweets table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS tweets (
                        tweet_id TEXT PRIMARY KEY,
                        conversation_id TEXT,
                        text TEXT,
                        author TEXT,
                        timestamp TIMESTAMP,
                        reply_to_id TEXT,
                        reply_to_user TEXT,
                        urls TEXT,
                        FOREIGN KEY (conversation_id) REFERENCES conversations (conversation_id)
                    )
                ''')
                
                # Create indices for better query performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_conv_time ON conversations (start_time)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_tweets_conv ON tweets (conversation_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_tweets_author ON tweets (author)')
                
                conn.commit()
                
        except sqlite3.Error as e:
            logger.error(f"Database initialization error: {str(e)}")
            raise

    def save_conversation(self, conversation_id: str, tweets: List[Tweet], metadata: ConversationMetadata) -> bool:
        """Save conversation to both JSON and SQLite storage."""
        try:
            # Save to JSON
            self._save_to_json(conversation_id, tweets, metadata)
            
            # Save to SQLite
            self._save_to_sqlite(conversation_id, tweets, metadata)
            
            self.stats['conversations_saved'] += 1
            self.stats['tweets_saved'] += len(tweets)
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving conversation {conversation_id}: {str(e)}")
            self.stats['storage_errors'] += 1
            return False

    def _save_to_json(self, conversation_id: str, tweets: List[Tweet], metadata: ConversationMetadata):
        """Save conversation to JSON file."""
        json_path = self.json_dir / f"conversation_{conversation_id}.json"
        
        conversation_data = {
            'conversation_id': conversation_id,
            'metadata': asdict(metadata),
            'messages': [asdict(tweet) for tweet in tweets]
        }
        
        # Convert datetime objects to strings
        conversation_data['metadata']['start_time'] = conversation_data['metadata']['start_time'].isoformat()
        conversation_data['metadata']['end_time'] = conversation_data['metadata']['end_time'].isoformat()
        
        for message in conversation_data['messages']:
            message['timestamp'] = message['timestamp'].isoformat()
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(conversation_data, f, ensure_ascii=False, indent=2)

    def _save_to_sqlite(self, conversation_id: str, tweets: List[Tweet], metadata: ConversationMetadata):
        """Save conversation to SQLite database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Save conversation metadata
                cursor.execute('''
                    INSERT OR REPLACE INTO conversations 
                    (conversation_id, start_time, end_time, participant_count, 
                     tweet_count, root_tweet_id, root_user)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    conversation_id,
                    metadata.start_time.isoformat(),
                    metadata.end_time.isoformat(),
                    metadata.participant_count,
                    metadata.tweet_count,
                    metadata.root_tweet_id,
                    metadata.root_user
                ))
                
                # Save tweets
                for tweet in tweets:
                    cursor.execute('''
                        INSERT OR REPLACE INTO tweets
                        (tweet_id, conversation_id, text, author, timestamp,
                         reply_to_id, reply_to_user, urls)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        tweet.tweet_id,
                        conversation_id,
                        tweet.text,
                        tweet.author,
                        tweet.timestamp.isoformat(),
                        tweet.reply_to_id,
                        tweet.reply_to_user,
                        json.dumps(tweet.urls)
                    ))
                
                conn.commit()
                
        except sqlite3.Error as e:
            logger.error(f"SQLite error saving conversation {conversation_id}: {str(e)}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Return storage statistics."""
        return self.stats
