"""Main entry point for the conversation pipeline."""

import os
import logging
from pathlib import Path
from datetime import datetime

from conversation_pipeline.core.pipeline import ConversationPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """
    Main function to process conversations from the tweet dataset.
    """
    try:
        # Input file configuration - using test file
        input_file = '/Users/mogen/Desktop/Research/data/df_test_10k.csv'
        
        # Output directory with timestamp
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'Results',
            f'conversations_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        )
        
        # Initialize and run the pipeline
        logger.info(f"Initializing conversation pipeline...")
        logger.info(f"Input file: {input_file}")
        logger.info(f"Output directory: {output_dir}")
        
        pipeline = ConversationPipeline(
            input_file=input_file,
            output_dir=output_dir
        )
        
        # Run with max_chunks=1 for testing
        success = pipeline.run(max_chunks=1)
        
        if success:
            logger.info("Pipeline completed successfully!")
            logger.info(f"Results saved to: {output_dir}")
        else:
            logger.error("Pipeline failed to complete successfully")
            
    except Exception as error:
        logger.error(f"Error in main: {str(error)}")
        raise

if __name__ == "__main__":
    main()
