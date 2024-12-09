import argparse
import logging
from pathlib import Path
from data_chunker import DataChunker
from reply_filter import ReplyFilter
from conversation_builder import ConversationBuilder
from conversation_storage import ConversationStorage

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_conversations(
    input_file: str,
    output_dir: str,
    chunk_size: int = 10000,
    min_conversation_length: int = 2
):
    """
    Process conversations from input file.
    
    Args:
        input_file: Path to input CSV file
        output_dir: Directory to store results
        chunk_size: Size of chunks to process
        min_conversation_length: Minimum length for valid conversations
    """
    try:
        # Initialize components
        chunker = DataChunker(chunk_size=chunk_size)
        reply_filter = ReplyFilter()
        conversation_builder = ConversationBuilder(min_conversation_length=min_conversation_length)
        storage = ConversationStorage(output_dir)
        
        logger.info(f"Starting conversation processing from {input_file}")
        
        # Process in chunks
        for chunk_num, chunk in enumerate(chunker.read_chunks(input_file)):
            logger.info(f"Processing chunk {chunk_num + 1}")
            
            # Filter replies
            replies = reply_filter.filter_replies(chunk)
            
            if len(replies) > 0:
                # Extract metadata
                metadata = reply_filter.extract_metadata(replies)
                logger.info(f"Chunk {chunk_num + 1} metadata: {metadata}")
                
                # Build conversations
                conversations = conversation_builder.build_conversations(replies)
                
                if conversations:
                    # Save conversations
                    batch_id = f"chunk_{chunk_num:04d}"
                    storage.save_conversations(conversations, batch_id)
                    
        logger.info("Conversation processing completed successfully")
        
    except Exception as e:
        logger.error(f"Error processing conversations: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Process conversations from Twitter data')
    parser.add_argument('input_file', help='Path to input CSV file')
    parser.add_argument('output_dir', help='Directory to store results')
    parser.add_argument('--chunk-size', type=int, default=10000, help='Size of chunks to process')
    parser.add_argument('--min-conversation-length', type=int, default=2,
                      help='Minimum number of posts to consider as a conversation')
    
    args = parser.parse_args()
    
    process_conversations(
        args.input_file,
        args.output_dir,
        args.chunk_size,
        args.min_conversation_length
    )

if __name__ == "__main__":
    main()
