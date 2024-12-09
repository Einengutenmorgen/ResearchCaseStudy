import logging
from typing import Dict, List, Optional
import openai
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PersonaManager:
    def __init__(self, api_key: str, cache_dir: Optional[str] = None):
        """
        Initialize the PersonaManager.
        
        Args:
            api_key: OpenAI API key
            cache_dir: Optional directory to cache persona results
        """
        self.client = openai.OpenAI(api_key=api_key)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.persona_cache = {}
        
    def generate_persona(self, user_id: str, posts: List[Dict]) -> Dict:
        """
        Generate or retrieve persona for a user.
        
        Args:
            user_id: User ID
            posts: List of user's posts
            
        Returns:
            Generated persona dictionary
        """
        # Check cache first
        if user_id in self.persona_cache:
            return self.persona_cache[user_id]
            
        # Check cache file if directory is specified
        if self.cache_dir:
            cache_file = self.cache_dir / f"persona_{user_id}.json"
            if cache_file.exists():
                with open(cache_file) as f:
                    return json.load(f)
        
        try:
            # Format posts for analysis
            post_texts = [post['full_text'] for post in posts]
            
            # Generate persona using OpenAI
            prompt = self._create_persona_prompt(post_texts)
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing social media behavior and creating detailed user personas."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            
            persona = {
                "user_id": user_id,
                "analysis": response.choices[0].message.content,
                "post_count": len(posts)
            }
            
            # Cache the result
            self.persona_cache[user_id] = persona
            if self.cache_dir:
                with open(cache_file, 'w') as f:
                    json.dump(persona, f, indent=2)
            
            return persona
            
        except Exception as e:
            logger.error(f"Error generating persona for user {user_id}: {str(e)}")
            return {"user_id": user_id, "analysis": "Error generating persona", "post_count": 0}
            
    def _create_persona_prompt(self, posts: List[str]) -> str:
        """Create prompt for persona generation."""
        return f"""Based on the following {len(posts)} social media posts, create a detailed persona of the user. 
Focus on their:
1. Communication style and tone
2. Typical topics and interests
3. Behavioral patterns
4. Values and beliefs
5. Interaction patterns

Posts:
{chr(10).join(f"- {post}" for post in posts)}

Provide a concise but comprehensive analysis that could help predict how this user would respond in conversations."""
