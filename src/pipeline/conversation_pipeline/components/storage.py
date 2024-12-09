import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConversationStorage:
    def __init__(self, base_dir: str):
        """
        Initialize the ConversationStorage.
        
        
        Args:
            base_dir: Base directory for storing conversations
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.base_dir / 'conversation_index.json'
        self.metadata_file = self.base_dir / 'metadata.json'
        
    def save_conversations(self, conversations: List[Dict[str, Any]], batch_id: str = None) -> str:
        """
        Save a batch of conversations.
        
        Args:
            conversations: List of conversation dictionaries
            batch_id: Optional batch identifier
            
        Returns:
            Path to saved batch
        """
        try:
            # Generate batch ID if not provided
            if batch_id is None:
                batch_id = datetime.now().strftime('%Y%m%d_%H%M%S')
                
            # Create batch directory
            batch_dir = self.base_dir / f'batch_{batch_id}'
            batch_dir.mkdir(exist_ok=True)
            
            # Save conversations
            conversation_paths = []
            for i, conv in enumerate(conversations):
                conv_file = batch_dir / f'conversation_{i:04d}.json'
                with open(conv_file, 'w') as f:
                    json.dump(conv, f, indent=2, default=str)
                conversation_paths.append(str(conv_file))
                
            # Update index
            self._update_index(batch_id, conversation_paths)
            
            # Save batch metadata
            self._save_batch_metadata(batch_id, conversations)
            
            logger.info(f"Saved {len(conversations)} conversations in batch {batch_id}")
            return str(batch_dir)
            
        except Exception as e:
            logger.error(f"Error saving conversations: {str(e)}")
            raise
            
    def load_conversations(self, batch_id: str = None) -> List[Dict[str, Any]]:
        """
        Load conversations from storage.
        
        Args:
            batch_id: Optional batch identifier to load specific batch
            
        Returns:
            List of conversation dictionaries
        """
        try:
            if not self.index_file.exists():
                raise FileNotFoundError("No conversation index found")
                
            with open(self.index_file) as f:
                index = json.load(f)
                
            if batch_id is not None:
                if batch_id not in index:
                    raise ValueError(f"Batch {batch_id} not found")
                paths = index[batch_id]
            else:
                # Load all conversations
                paths = [path for batch_paths in index.values() for path in batch_paths]
                
            conversations = []
            for path in paths:
                with open(path) as f:
                    conversations.append(json.load(f))
                    
            logger.info(f"Loaded {len(conversations)} conversations")
            return conversations
            
        except Exception as e:
            logger.error(f"Error loading conversations: {str(e)}")
            raise
            
    def _update_index(self, batch_id: str, conversation_paths: List[str]):
        """
        Update the conversation index.
        
        Args:
            batch_id: Batch identifier
            conversation_paths: List of paths to conversation files
        """
        index = {}
        if self.index_file.exists():
            with open(self.index_file) as f:
                index = json.load(f)
                
        index[batch_id] = conversation_paths
        
        with open(self.index_file, 'w') as f:
            json.dump(index, f, indent=2)
            
    def _save_batch_metadata(self, batch_id: str, conversations: List[Dict[str, Any]]):
        """
        Save metadata for a batch of conversations.
        
        Args:
            batch_id: Batch identifier
            conversations: List of conversation dictionaries
        """
        metadata = {
            'batch_id': batch_id,
            'timestamp': datetime.now().isoformat(),
            'conversation_count': len(conversations),
            'total_messages': sum(len(c['messages']) for c in conversations),
            'total_participants': len(set(
                p for c in conversations 
                for p in c['participants']
            )),
            'average_conversation_length': sum(len(c['messages']) for c in conversations) / len(conversations)
        }
        
        # Load existing metadata
        all_metadata = {}
        if self.metadata_file.exists():
            with open(self.metadata_file) as f:
                all_metadata = json.load(f)
                
        all_metadata[batch_id] = metadata
        
        with open(self.metadata_file, 'w') as f:
            json.dump(all_metadata, f, indent=2)
