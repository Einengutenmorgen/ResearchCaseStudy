"""Module for analyzing user posts to extract patterns and characteristics."""

import logging
from typing import Dict, List, Optional
from collections import Counter
import re
from datetime import datetime
import emoji

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PostAnalyzer:
    """Class responsible for analyzing user posts to extract patterns and characteristics."""
    
    def __init__(self):
        """Initialize the PostAnalyzer."""
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.mention_pattern = re.compile(r'@\w+')
        self.hashtag_pattern = re.compile(r'#\w+')

    def analyze_posts(self, posts: List[Dict]) -> Dict:
        """
        Analyze a collection of user posts to extract patterns and characteristics.
        
        Args:
            posts: List of post dictionaries containing at least 'full_text' and 'created_at'
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            post_texts = [post['full_text'] for post in posts]
            
            # Handle both string and datetime objects for created_at
            timestamps = []
            for post in posts:
                created_at = post['created_at']
                if isinstance(created_at, str):
                    # Handle various datetime string formats
                    try:
                        # Try ISO format first
                        ts = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    except ValueError:
                        try:
                            # Try pandas timestamp format
                            ts = datetime.strptime(created_at, '%Y-%m-%d %H:%M:%S%z')
                        except ValueError:
                            # Default to just date if time parsing fails
                            ts = datetime.strptime(created_at.split()[0], '%Y-%m-%d')
                else:
                    # Already a datetime object
                    ts = created_at
                timestamps.append(ts)

            analysis = {
                "post_metrics": self._analyze_post_metrics(post_texts),
                "temporal_patterns": self._analyze_temporal_patterns(timestamps),
                "content_patterns": self._analyze_content_patterns(post_texts),
                "interaction_patterns": self._analyze_interaction_patterns(post_texts),
                "language_patterns": self._analyze_language_patterns(post_texts)
            }
            
            logger.info(f"Successfully analyzed {len(posts)} posts")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing posts: {str(e)}")
            return {"error": str(e)}

    def _analyze_post_metrics(self, posts: List[str]) -> Dict:
        """Analyze basic metrics of posts."""
        word_counts = [len(self._clean_text(post).split()) for post in posts]
        return {
            "total_posts": len(posts),
            "avg_word_count": sum(word_counts) / len(posts) if posts else 0,
            "max_word_count": max(word_counts) if posts else 0,
            "min_word_count": min(word_counts) if posts else 0
        }

    def _analyze_temporal_patterns(self, timestamps: List[datetime]) -> Dict:
        """Analyze posting time patterns."""
        if not timestamps:
            return {}
            
        hour_distribution = Counter([ts.hour for ts in timestamps])
        weekday_distribution = Counter([ts.weekday() for ts in timestamps])
        
        return {
            "most_active_hours": sorted(hour_distribution.items(), 
                                     key=lambda x: x[1], 
                                     reverse=True)[:3],
            "most_active_days": sorted(weekday_distribution.items(), 
                                    key=lambda x: x[1], 
                                    reverse=True),
            "posting_frequency": len(timestamps) / ((max(timestamps) - min(timestamps)).days + 1)
        }

    def _analyze_content_patterns(self, posts: List[str]) -> Dict:
        """Analyze patterns in post content."""
        url_counts = sum(bool(self.url_pattern.search(post)) for post in posts)
        emoji_counts = sum(bool(emoji.emoji_count(post)) for post in posts)
        hashtags = [tag for post in posts 
                   for tag in self.hashtag_pattern.findall(post)]
        
        return {
            "url_frequency": url_counts / len(posts) if posts else 0,
            "emoji_frequency": emoji_counts / len(posts) if posts else 0,
            "top_hashtags": Counter(hashtags).most_common(5)
        }

    def _analyze_interaction_patterns(self, posts: List[str]) -> Dict:
        """Analyze interaction patterns in posts."""
        mentions = [mention for post in posts 
                   for mention in self.mention_pattern.findall(post)]
        
        return {
            "mention_frequency": len(mentions) / len(posts) if posts else 0,
            "unique_mentions": len(set(mentions)),
            "top_mentioned_users": Counter(mentions).most_common(5)
        }

    def _analyze_language_patterns(self, posts: List[str]) -> Dict:
        """Analyze language patterns in posts."""
        cleaned_posts = [self._clean_text(post) for post in posts]
        words = [word for post in cleaned_posts 
                for word in post.split()]
        
        return {
            "vocabulary_size": len(set(words)),
            "avg_post_length": sum(len(post) for post in posts) / len(posts) if posts else 0,
            "common_words": Counter(words).most_common(10)
        }

    def _clean_text(self, text: str) -> str:
        """Clean text by removing URLs, mentions, and extra whitespace."""
        text = self.url_pattern.sub('', text)
        text = self.mention_pattern.sub('', text)
        text = self.hashtag_pattern.sub('', text)
        return ' '.join(text.split())
