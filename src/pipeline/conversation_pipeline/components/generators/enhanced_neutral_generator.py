import logging
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from openai import OpenAI
import os
from enum import Enum

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DescriptionStyle(Enum):
    CONCISE = "concise"
    DETAILED = "detailed"
    ACADEMIC = "academic"

class OutputFormat(Enum):
    PLAIN = "plain"
    STRUCTURED = "structured"
    JSON = "json"

@dataclass
class GeneratorConfig:
    # Model Configuration
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.3
    max_tokens: int = 16384
    
    # Prompt Customization
    description_style: DescriptionStyle = DescriptionStyle.CONCISE
    focus_aspects: List[str] = None
    system_prompt_template: Optional[str] = None
    
    # Processing Parameters
    batch_size: int = 5
    retry_attempts: int = 3
    timeout: int = 45
    
    # Output Configuration
    output_format: OutputFormat = OutputFormat.PLAIN
    include_metadata: bool = False
    language: str = "en"
    
    # Filtering Parameters
    min_post_length: int = 1
    max_post_length: int = 1000
    content_filters: List[str] = None

    def __post_init__(self):
        if self.focus_aspects is None:
            self.focus_aspects = ["tone", "topic", "intent"]
        if self.content_filters is None:
            self.content_filters = []

class EnhancedNeutralDescriptionGenerator:
    """Enhanced version of NeutralDescriptionGenerator with additional configuration options."""
    
    def __init__(self, config: Optional[GeneratorConfig] = None):
        """
        Initialize the generator with configuration.
        
        Args:
            config: Configuration object with parameters for the generator
        """
        self.config = config or GeneratorConfig()
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
    def _create_system_prompt(self) -> str:
        """Create system prompt based on configuration."""
        if self.config.system_prompt_template:
            return self.config.system_prompt_template
            
        style_instructions = {
            DescriptionStyle.CONCISE: "Create brief, to-the-point descriptions",
            DescriptionStyle.DETAILED: "Provide comprehensive, detailed analysis",
            DescriptionStyle.ACADEMIC: "Use formal, academic language in descriptions"
        }
        
        focus_aspects_str = ", ".join(self.config.focus_aspects)
        
        return f"""You are a neutral observer tasked with creating {self.config.description_style.value} descriptions of social media posts.

Focus on the following aspects: {focus_aspects_str}
Style Instruction: {style_instructions[self.config.description_style]}

Additional Guidelines:
- Use {self.config.language} language
- Maintain neutral, unbiased language
- Focus on content structure and intent
- Ignore platform-specific features unless crucial
- Do NOT qoute or repeat any parts of the post
"""

    def create_neutral_prompt(self, post: str) -> List[Dict[str, str]]:
        """
        Create a prompt for neutral post description.
        
        Args:
            post: The social media post to describe
            
        Returns:
            List of message dictionaries for the OpenAI API
        """
        system_message = {
            "role": "system",
            "content": self._create_system_prompt()
        }
        
        user_message = {
            "role": "user",
            "content": f"Create a neutral description of this social media post: {post}"
        }
        
        return [system_message, user_message]

    def _validate_post(self, post: str) -> bool:
        """
        Validate post against filtering parameters.
        
        Args:
            post: The post to validate
            
        Returns:
            bool: Whether the post passes validation
        """
        if not self.config.min_post_length <= len(post) <= self.config.max_post_length:
            return False
            
        for filter_term in self.config.content_filters:
            if filter_term.lower() in post.lower():
                return False
                
        return True

    def _format_output(self, description: str, post: str) -> Union[str, Dict]:
        """
        Format the output according to configuration.
        
        Args:
            description: The generated description
            post: The original post
            
        Returns:
            Formatted output according to output_format configuration
        """
        if self.config.output_format == OutputFormat.PLAIN:
            return description
            
        base_output = {
            "description": description,
            "style": self.config.description_style.value,
            "focus_aspects": self.config.focus_aspects
        }
        
        if self.config.include_metadata:
            base_output.update({
                "original_post": post,
                "post_length": len(post),
                "description_length": len(description)
            })
            
        if self.config.output_format == OutputFormat.JSON:
            return base_output
        
        # Structured format
        return "\n".join([f"{k}: {v}" for k, v in base_output.items()])

    async def generate_descriptions_batch(self, posts: List[str]) -> List[Union[str, Dict]]:
        """
        Generate neutral descriptions for a batch of posts.
        
        Args:
            posts: List of posts to process
            
        Returns:
            List of descriptions in the configured format
        """
        results = []
        valid_posts = [post for post in posts if self._validate_post(post)]
        
        for i in range(0, len(valid_posts), self.config.batch_size):
            batch = valid_posts[i:i + self.config.batch_size]
            batch_results = []
            
            for post in batch:
                for attempt in range(self.config.retry_attempts):
                    try:
                        prompt = self.create_neutral_prompt(post)
                        response = self.openai_client.chat.completions.create(
                            model=self.config.model_name,
                            messages=prompt,
                            temperature=self.config.temperature,
                            max_tokens=self.config.max_tokens,
                            timeout=self.config.timeout
                        )
                        description = response.choices[0].message.content.strip()
                        batch_results.append(self._format_output(description, post))
                        break
                    except Exception as e:
                        if attempt == self.config.retry_attempts - 1:
                            logger.error(f"Failed to process post after {self.config.retry_attempts} attempts: {e}")
                            batch_results.append(None)
                        else:
                            logger.warning(f"Attempt {attempt + 1} failed, retrying...")
                            
            results.extend(batch_results)
            
        return results

    def generate_description(self, post: str) -> Optional[str]:
        """Generate a neutral description for a single post."""
        try:
            messages = [
                {
                    "role": "system",
                    "content": self._create_system_prompt()
                },
                {
                    "role": "user",
                    "content": f"Post: {post}\n\nGenerate a neutral description of this post."
                }
            ]

            response = self.openai_client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.timeout
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating description: {e}")
            return None
