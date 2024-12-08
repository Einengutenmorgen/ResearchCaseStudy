import argparse
import logging
from pathlib import Path
from root_tweet_fetcher import RootTweetFetcher

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Fetch root tweets for conversations')
    parser.add_argument('data_file', help='Path to the original data CSV file')
    parser.add_argument('conversation_dir', help='Directory containing conversation JSON files')
    parser.add_argument('--output-dir', help='Optional output directory for enhanced conversations')
    
    args = parser.parse_args()
    
    try:
        # Initialize fetcher
        fetcher = RootTweetFetcher(args.data_file)
        
        # Process conversations
        stats = fetcher.process_conversation_batch(
            args.conversation_dir,
            args.output_dir
        )
        
        logger.info("Root tweet fetching completed successfully")
        logger.info(f"Final statistics: {stats}")
        
    except Exception as e:
        logger.error(f"Error in root tweet fetching: {str(e)}")
        raise

if __name__ == "__main__":
    main()
