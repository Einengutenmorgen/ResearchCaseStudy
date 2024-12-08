import pandas as pd
import logging
from typing import List, Dict, Any
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConversationBuilder:
    def __init__(self, min_conversation_length: int = 2):
        """
        Initialize the ConversationBuilder.
        
        Args:
            min_conversation_length: Minimum number of posts to consider as a conversation
        """
        self.min_conversation_length = min_conversation_length
        
    def build_conversations(self, replies: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Build conversation threads from replies.
        
        Args:
            replies: DataFrame of filtered replies
            
        Returns:
            List of conversation dictionaries
        """
        try:
            # Create a mapping of post ID to its replies
            reply_map = defaultdict(list)
            for _, row in replies.iterrows():
                reply_map[row['reply_to_id']].append(row.to_dict())
            
            # Find root posts (those that started conversations)
            root_posts = self._identify_root_posts(replies)
            
            # Build conversation trees
            conversations = []
            for root_id in root_posts:
                conversation = self._build_conversation_tree(root_id, reply_map)
                if self._is_valid_conversation(conversation):
                    conversations.append(conversation)
            
            logger.info(f"Built {len(conversations)} valid conversations")
            return conversations
            
        except Exception as e:
            logger.error(f"Error building conversations: {str(e)}")
            raise
            
    def _identify_root_posts(self, replies: pd.DataFrame) -> List[int]:
        """
        Identify posts that started conversations.
        
        Args:
            replies: DataFrame of filtered replies
            
        Returns:
            List of root post IDs
        """
        # Find posts that were replied to but are not themselves replies
        all_reply_targets = set(replies['reply_to_id'].unique())
        all_posts = set(replies['tweet_id'])
        
        return list(all_reply_targets - all_posts)
        
    def _build_conversation_tree(self, root_id: int, reply_map: Dict) -> Dict[str, Any]:
        """
        Build a conversation tree starting from a root post.
        
        Args:
            root_id: ID of the root post
            reply_map: Mapping of post IDs to their replies
            
        Returns:
            Dictionary representing the conversation
        """
        conversation = {
            'root_id': root_id,
            'messages': [],
            'participants': set(),
            'metadata': {}
        }
        
        # Build the conversation tree using DFS
        self._add_replies_recursive(root_id, reply_map, conversation)
        
        # Add metadata
        conversation['metadata'] = {
            'length': len(conversation['messages']),
            'participant_count': len(conversation['participants']),
            'time_span': self._calculate_time_span(conversation['messages'])
        }
        
        # Convert participants set to list for JSON serialization
        conversation['participants'] = list(conversation['participants'])
        
        return conversation
        
    def _add_replies_recursive(self, post_id: int, reply_map: Dict, conversation: Dict):
        """
        Recursively add replies to the conversation.
        
        Args:
            post_id: Current post ID
            reply_map: Mapping of post IDs to their replies
            conversation: Conversation dictionary to update
        """
        for reply in reply_map.get(post_id, []):
            conversation['messages'].append(reply)
            conversation['participants'].add(reply['original_user_id'])
            self._add_replies_recursive(reply['tweet_id'], reply_map, conversation)
            
    def _is_valid_conversation(self, conversation: Dict) -> bool:
        """
        Check if a conversation meets the minimum requirements.
        
        Args:
            conversation: Conversation dictionary
            
        Returns:
            True if valid, False otherwise
        """
        return len(conversation['messages']) >= self.min_conversation_length
        
    def _calculate_time_span(self, messages: List[Dict]) -> Dict:
        """
        Calculate the time span of a conversation.
        
        Args:
            messages: List of messages in the conversation
            
        Returns:
            Dictionary with start and end times
        """
        if not messages:
            return {'start': None, 'end': None}
            
        times = [pd.to_datetime(msg['created_at']) for msg in messages]
        return {
            'start': min(times),
            'end': max(times),
            'duration_seconds': (max(times) - min(times)).total_seconds()
        }
