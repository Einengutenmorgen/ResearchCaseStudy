import pandas as pd
from typing import List

class PromptFormatter:
    """Formats prompts for OpenAI analysis."""
    
    @staticmethod
    def create_detailed_prompt(row: pd.Series, posts_for_persona: List[str]) -> str:
        """Create a detailed prompt for persona analysis."""
        # Format the tweets as a numbered list
        tweets_text = "\n".join([f"{i+1}. {tweet}" for i, tweet in enumerate(posts_for_persona)])
        
        prompt = f"""Analyze the following social media posts from a single user and create a detailed character description.
Focus on their personality traits, communication style, interests, and behavioral patterns.
Consider their humor style, emotional expressions, values, and social interaction patterns.

User's Posts:
{tweets_text}

Based on these posts, provide a comprehensive character description covering:
1. Humor Style
2. Communication Patterns
3. Emotional Expressions
4. Values and Beliefs
5. Interests and Hobbies
6. Social Interactions
7. Personality Traits
8. Cultural Background

Organize your analysis with clear sections and specific examples from the posts."""
        
        return prompt
