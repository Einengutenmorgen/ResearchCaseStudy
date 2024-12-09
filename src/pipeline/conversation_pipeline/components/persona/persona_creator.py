"""Module for creating user personas based on their posts."""

import logging
from typing import Dict, List, Optional
from pathlib import Path
import json
import openai
from openai import OpenAI
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PersonaCreator:
    """Class responsible for creating user personas based on their posts."""
    
    def __init__(self, api_key: str, cache_dir: Optional[str] = None):
        """
        Initialize the PersonaCreator.
        
        Args:
            api_key: OpenAI API key
            cache_dir: Optional directory to cache persona results
        """
        self.client = OpenAI(api_key=api_key)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.persona_cache = {}

    def create_persona(self, user_id: str, posts: List[Dict]) -> Dict:
        """
        Create or retrieve a persona for a user based on their posts.
        
        Args:
            user_id: User ID
            posts: List of user's posts
            
        Returns:
            Generated persona dictionary containing analysis and metadata
        """
        # Check memory cache first
        if user_id in self.persona_cache:
            logger.info(f"Retrieved persona for user {user_id} from memory cache")
            return self.persona_cache[user_id]
            
        # Check file cache if directory is specified
        if self.cache_dir:
            cache_file = self.cache_dir / f"persona_{user_id}.json"
            if cache_file.exists():
                with open(cache_file) as f:
                    logger.info(f"Retrieved persona for user {user_id} from file cache")
                    return json.load(f)
    
        try:
            # Format posts for analysis
            post_texts = [post['full_text'] for post in posts]
            
            # Generate persona using OpenAI
            prompt = self._create_persona_prompt(post_texts)
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert at analyzing social media behavior and creating detailed user personas. "
                                 "Focus on communication style, interests, values, and interaction patterns."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            
            persona = {
                "user_id": user_id,
                "analysis": response.choices[0].message.content,
                "metadata": {
                    "post_count": len(posts),
                    "analysis_timestamp": str(datetime.now()),
                    "model_used": "gpt-4o"
                }
            }
            
            # Cache the result
            self._cache_persona(user_id, persona)
            
            logger.info(f"Successfully created persona for user {user_id}")
            return persona
            
        except Exception as e:
            logger.error(f"Error generating persona for user {user_id}: {str(e)}")
            return {
                "user_id": user_id, 
                "analysis": "Error generating persona", 
                "metadata": {
                    "error": str(e),
                    "post_count": len(posts)
                }
            }

    def _create_persona_prompt(self, posts: List[str]) -> str:
        """
        Create a detailed prompt for persona generation.
        
        Args:
            posts: List of user's post texts
            
        Returns:
            Formatted prompt string
        """
        return f"""Based on the following {len(posts)} social media posts, create a detailed persona of the user. 
Focus on:
1. Communication Style
   - Writing style and tone
   - Language patterns and preferences
   - Level of formality

2. Topics and Interests
   - Main discussion topics
   - Areas of expertise
   - Recurring themes

3. Behavioral Patterns
   - Posting frequency and timing
   - Response patterns
   - Content sharing habits

4. Values and Beliefs
   - Expressed opinions
   - Core values
   - Worldview indicators

5. Interaction Style
   - Engagement with others
   - Community involvement
   - Response to different topics

Posts:
{chr(10).join(f"- {post}" for post in posts)}

Provide a structured analysis that captures the user's distinct characteristics and behavioral patterns."""

    def _cache_persona(self, user_id: str, persona: Dict) -> None:
        """
        Cache the generated persona both in memory and file system if configured.
        
        Args:
            user_id: User ID
            persona: Generated persona dictionary
        """
        # Cache in memory
        self.persona_cache[user_id] = persona
        
        # Cache to file if directory is specified
        if self.cache_dir:
            cache_file = self.cache_dir / f"persona_{user_id}.json"
            with open(cache_file, 'w') as f:
                json.dump(persona, f, indent=2)
            logger.debug(f"Cached persona for user {user_id} to {cache_file}")
