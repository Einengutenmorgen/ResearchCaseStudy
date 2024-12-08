import os
import logging
from pathlib import Path
from typing import Optional
import pandas as pd
from datetime import datetime

from conversation_pipeline import ConversationExtractor
from conversation_storage import ConversationStorageManager

logger = logging.getLogger(__name__)

class ConversationPipeline:
    def __init__(self, input_file: str, output_dir: Optional[str] = None):
        """
        Initialize the conversation pipeline.
        
        Args:
            input_file: Path to input CSV file
            output_dir: Optional output directory path
        """
        self.input_file = Path(input_file)
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
            
        self.output_dir = Path(output_dir or f"Results/conversations_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.extractor = ConversationExtractor()
        self.storage = ConversationStorageManager(str(self.output_dir))
        
        # Configuration
        self.chunk_size = 10000  # Adjust based on available memory
        
    def run(self, max_chunks: Optional[int] = None) -> bool:
        """
        Run the conversation extraction pipeline.
        
        Args:
            max_chunks: Optional maximum number of chunks to process (for testing)
        
        Returns:
            bool: True if processing completed successfully
        """
        try:
            logger.info(f"Starting conversation extraction from {self.input_file}")
            
            # Process the CSV file in chunks
            chunks_processed = 0
            for chunk in pd.read_csv(self.input_file, chunksize=self.chunk_size):
                # Process chunk
                self.extractor.process_chunk(chunk)
                
                # Save completed conversations
                self._save_completed_conversations()
                
                chunks_processed += 1
                if max_chunks and chunks_processed >= max_chunks:
                    logger.info(f"Reached maximum chunks limit ({max_chunks})")
                    break
            
            # Final save of any remaining conversations
            self._save_completed_conversations(final=True)
            
            # Log final statistics
            self._log_statistics()
            
            return True
            
        except Exception as e:
            logger.error(f"Error in conversation pipeline: {str(e)}")
            return False
    
    def _save_completed_conversations(self, final: bool = False):
        """Save completed conversations to storage."""
        for conv_id, tweets in self.extractor.conversations.items():
            # Only save if conversation is complete or this is the final save
            if final or self._is_conversation_complete(tweets):
                if self.extractor.validate_conversation(conv_id, tweets):
                    metadata = self.extractor.get_conversation_metadata(conv_id, tweets)
                    self.storage.save_conversation(conv_id, tweets, metadata)
    
    def _is_conversation_complete(self, tweets):
        """
        Check if a conversation appears to be complete.
        This is a heuristic based on time gaps between tweets.
        """
        if not tweets:
            return False
            
        # Sort tweets by timestamp
        sorted_tweets = sorted(tweets, key=lambda x: x.timestamp)
        
        # If the last tweet is more than 24 hours old, consider the conversation complete
        last_tweet_time = sorted_tweets[-1].timestamp
        current_time = pd.Timestamp.now(tz='UTC')
        time_since_last = current_time - last_tweet_time
        
        return time_since_last.total_seconds() > 24 * 3600
    
    def _log_statistics(self):
        """Log final statistics from both extractor and storage."""
        extractor_stats = self.extractor.get_stats()
        storage_stats = self.storage.get_stats()
        
        logger.info("Pipeline Statistics:")
        logger.info(f"Processed chunks: {extractor_stats['processed_chunks']}")
        logger.info(f"Processed tweets: {extractor_stats['processed_tweets']}")
        logger.info(f"Conversations found: {extractor_stats['conversations_found']}")
        logger.info(f"Conversations saved: {storage_stats['conversations_saved']}")
        logger.info(f"Tweets saved: {storage_stats['tweets_saved']}")
        logger.info(f"Storage errors: {storage_stats['storage_errors']}")

def main():
    """Main entry point for running the conversation pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract conversations from tweet data')
    parser.add_argument('input_file', help='Path to input CSV file')
    parser.add_argument('--output-dir', help='Output directory (optional)')
    parser.add_argument('--max-chunks', type=int, help='Maximum number of chunks to process (optional)')
    
    args = parser.parse_args()
    
    pipeline = ConversationPipeline(args.input_file, args.output_dir)
    success = pipeline.run(max_chunks=args.max_chunks)
    
    if success:
        logger.info("Pipeline completed successfully")
    else:
        logger.error("Pipeline failed")
        exit(1)

if __name__ == "__main__":
    main()
