"""Core pipeline implementation."""

import os
import logging
from pathlib import Path
from typing import Optional
import pandas as pd
from datetime import datetime

from ..components.storage import ConversationStorage
from ..components.builder import ConversationBuilder
from ..components.reply_filter import ReplyFilter
from ..utils.data_chunker import DataChunker
from ..components.analyzers.openai_analyzer import OpenAIAnalyzer
from ..components.analyzers.similarity_analyzer import SimilarityAnalyzer
from ..components.analyzers.rouge_evaluator import RougeEvaluator
from ..components.generators.enhanced_neutral_generator import EnhancedNeutralDescriptionGenerator

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
        self.builder = ConversationBuilder()
        self.storage = ConversationStorage(str(self.output_dir))
        self.reply_filter = ReplyFilter()
        self.chunker = DataChunker()
        
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
                if max_chunks and chunks_processed >= max_chunks:
                    break
                    
                logger.info(f"Processing chunk {chunks_processed + 1}")
                
                # Filter replies
                replies = self.reply_filter.filter_replies(chunk)
                
                if len(replies) > 0:
                    # Build conversations
                    conversations = self.builder.build_conversations(replies)
                    
                    if conversations:
                        # Save conversations
                        batch_id = f"chunk_{chunks_processed:04d}"
                        self.storage.save_conversations(conversations, batch_id)
                
                chunks_processed += 1
                
            logger.info("Conversation extraction completed successfully")
            return True
            
        except Exception as error:
            logger.error(f"Error in pipeline execution: {str(error)}")
            return False
