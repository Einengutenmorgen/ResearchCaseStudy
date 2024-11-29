import os
import logging
from typing import Optional, Dict
from openai import OpenAI
from datetime import datetime
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PostRegenerator:
    """Handles the regeneration of posts based on persona descriptions and neutral content descriptions."""
    
    def __init__(self, model_name: str = "gpt-4o"):
        """
        Initialize the PostRegenerator.
        
        Args:
            model_name: The OpenAI model to use for generation
        """
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.model_name = model_name
    
    def create_regeneration_prompt(self, persona: str, neutral_description: str) -> str:
        """
        Create a prompt for post regeneration.
        
        Args:
            persona: The persona description
            neutral_description: The neutral description of the content to generate
            
        Returns:
            str: The formatted prompt
        """
        return f"""As a social media post author with the following characteristics:

{persona}

Create a social media post that expresses this content:
{neutral_description}

Important:
- Match the writing style, tone, and characteristics of the persona
- Maintain the core message from the neutral description
- Keep the post within 280 characters
- Make it feel authentic to the persona

Generate only the post text, without any additional explanation."""

    def regenerate_post(self, persona: str, neutral_description: str) -> Optional[str]:
        """
        Generate a new post based on the persona and neutral description.
        
        Args:
            persona: The persona description
            neutral_description: The neutral description of the content
            
        Returns:
            Optional[str]: The generated post or None if generation fails
        """
        try:
            prompt = self.create_regeneration_prompt(persona, neutral_description)
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert at mimicking writing styles and generating authentic social media posts."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=100  # Limiting tokens for social media post length
            )
            
            generated_post = response.choices[0].message.content.strip()
            logger.info("Successfully generated new post")
            return generated_post
            
        except Exception as e:
            logger.error(f"Error generating post: {e}")
            return None

class RegenerationResultsManager:
    """Manages the storage and organization of regenerated post results."""
    
    def __init__(self, output_dir: str):
        """
        Initialize the results manager.
        
        Args:
            output_dir: Directory to store results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def save_results(self, results: list[Dict]) -> str:
        """
        Save regeneration results to a CSV file.
        
        Args:
            results: List of dictionaries containing regeneration results
            
        Returns:
            str: Path to the saved file
        """
        try:
            df = pd.DataFrame(results)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"regenerated_posts_{timestamp}.csv"
            filepath = os.path.join(self.output_dir, filename)
            
            df.to_csv(filepath, index=False)
            logger.info(f"Saved regeneration results to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise