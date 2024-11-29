import os
import logging
from typing import List, Dict
from openai import OpenAI
import tiktoken
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class OpenAIAnalyzer:
    def __init__(self, api_key: str = None, model_name: str = "gpt-4o"):
        """Initialize OpenAI client with API key."""
        self.client = OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
        self.model_name = model_name
        self.encoding = tiktoken.encoding_for_model(model_name)
        self.max_context_tokens = 8192  # GPT-4's context window
        
    def count_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Count the number of tokens in the messages."""
        num_tokens = 0
        for message in messages:
            # Count tokens in the content
            num_tokens += len(self.encoding.encode(message["content"]))
            # Add overhead for message format
            num_tokens += 4  # Every message follows <im_start>{role/name}\n{content}<im_end>\n
        num_tokens += 2  # Every reply is primed with <im_start>assistant
        return num_tokens
        
    def analyze_persona(self, prompt: str) -> str:
        """Analyze user persona using OpenAI."""
        try:
            messages = [
                {"role": "system", "content": "You are an expert at analyzing social media posts and creating detailed personality profiles."},
                {"role": "user", "content": prompt}
            ]
            
            # Count tokens before making the API call
            token_count = self.count_tokens(messages)
            logger.info(f"Total tokens in prompt: {token_count}")
            
            # Calculate maximum completion tokens to stay within limits
            max_completion_tokens = min(
                self.max_context_tokens - token_count - 100,  # Leave 100 tokens as buffer
                4096  # Maximum response tokens for GPT-4
            )
            
            if max_completion_tokens <= 0:
                logger.error(f"Prompt too long: {token_count} tokens")
                return None
            
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.3,
                max_tokens=max_completion_tokens,
                timeout=45  # Using timeout instead of request_timeout
            )
            return completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in OpenAI API call: {e}")
            return None
