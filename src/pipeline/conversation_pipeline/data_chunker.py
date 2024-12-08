import pandas as pd
import logging
from typing import Iterator, Optional
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataChunker:
    def __init__(self, chunk_size: int = 10000):
        """
        Initialize the DataChunker.
        
        Args:
            chunk_size: Number of rows to process at once
        """
        self.chunk_size = chunk_size
        
    def read_chunks(self, file_path: str, columns: Optional[list] = None) -> Iterator[pd.DataFrame]:
        """
        Read data in chunks from a CSV file.
        
        Args:
            file_path: Path to the input CSV file
            columns: Optional list of columns to read
            
        Yields:
            DataFrame chunks
        """
        try:
            total_rows = sum(1 for _ in open(file_path)) - 1  # -1 for header
            logger.info(f"Processing file with {total_rows} rows")
            
            chunks_iterator = pd.read_csv(
                file_path,
                chunksize=self.chunk_size,
                usecols=columns
            )
            
            for chunk_num, chunk in enumerate(chunks_iterator):
                processed_rows = min((chunk_num + 1) * self.chunk_size, total_rows)
                logger.info(f"Processing chunk {chunk_num + 1}, rows {processed_rows}/{total_rows}")
                yield chunk
                
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
            raise
            
    def save_intermediate(self, chunk: pd.DataFrame, output_dir: str, chunk_num: int) -> str:
        """
        Save intermediate chunk results.
        
        Args:
            chunk: DataFrame chunk to save
            output_dir: Directory to save the chunk
            chunk_num: Chunk number for filename
            
        Returns:
            Path to saved file
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            file_path = output_path / f"chunk_{chunk_num:04d}.csv"
            chunk.to_csv(file_path, index=False)
            logger.info(f"Saved intermediate chunk to {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Error saving intermediate chunk: {str(e)}")
            raise
