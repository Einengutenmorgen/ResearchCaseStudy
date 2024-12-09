import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import os
from dataclasses import dataclass
from enum import Enum

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimilarityMetric(Enum):
    """Enum for different similarity metrics."""
    SEMANTIC = "semantic"
    COSINE = "cosine"
    LLM = "llm"
    COMBINED = "combined"

@dataclass
class SimilarityScore:
    """Data class to hold similarity scores."""
    semantic_similarity: float
    cosine_similarity: float
    llm_similarity: Optional[float] = None
    combined_score: Optional[float] = None
    llm_explanation: Optional[str] = None

class SimilarityAnalyzer:
    """Analyzes similarity between original and regenerated posts using multiple metrics."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the SimilarityAnalyzer.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.sentence_transformer = SentenceTransformer(model_name)
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity using sentence transformers.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            float: Semantic similarity score
        """
        try:
            embeddings1 = self.sentence_transformer.encode([text1])
            embeddings2 = self.sentence_transformer.encode([text2])
            similarity = cosine_similarity(embeddings1, embeddings2)[0][0]
            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {e}")
            return 0.0
            
    def calculate_cosine_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between TF-IDF vectors of the texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            float: Cosine similarity score
        """
        try:
            # Use sentence transformers for consistent embeddings
            embeddings1 = self.sentence_transformer.encode([text1])
            embeddings2 = self.sentence_transformer.encode([text2])
            similarity = cosine_similarity(embeddings1, embeddings2)[0][0]
            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0
            
    def get_llm_similarity_judgment(self, original: str, regenerated: str, neutral: str) -> Tuple[float, str]:
        """
        Get similarity judgment from LLM comparing original and regenerated posts.
        
        Args:
            original: Original post
            regenerated: Regenerated post
            neutral: Neutral description used for regeneration
            
        Returns:
            Tuple[float, str]: Similarity score and explanation
        """
        try:
            prompt = f"""Analyze the similarity between an original social media post and its regenerated version.
            Consider these aspects:
            1. Core message and intent
            2. Tone and style
            3. Key information preserved
            4. Emotional impact
            
            Original post: "{original}"
            Neutral description: "{neutral}"
            Regenerated post: "{regenerated}"
            
            Provide:
            1. A similarity score from 0.0 to 1.0
            2. A brief explanation of the score
            
            Format: score|explanation
            Example: 0.85|Good preservation of core message with similar tone, though slightly different wording
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing text similarity and writing styles."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=100
            )
            
            result = response.choices[0].message.content.strip()
            score_str, explanation = result.split('|', 1)
            score = float(score_str)
            
            return score, explanation
            
        except Exception as e:
            logger.error(f"Error getting LLM similarity judgment: {e}")
            return 0.0, "Error in LLM analysis"
            
    def calculate_combined_score(self, scores: SimilarityScore) -> float:
        """
        Calculate a weighted combined score from multiple similarity metrics.
        
        Args:
            scores: SimilarityScore object containing individual scores
            
        Returns:
            float: Combined weighted score
        """
        weights = {
            'semantic': 0.3,
            'cosine': 0.3,
            'llm': 0.4
        }
        
        total_score = (
            scores.semantic_similarity * weights['semantic'] +
            scores.cosine_similarity * weights['cosine']
        )
        
        if scores.llm_similarity is not None:
            total_score += scores.llm_similarity * weights['llm']
        else:
            # Redistribute LLM weight to other metrics if LLM score is not available
            weight_factor = 1 / (1 - weights['llm'])
            total_score *= weight_factor
            
        return total_score
        
    def analyze_similarity(
        self,
        original: str,
        regenerated: str,
        neutral: str,
        metrics: Optional[List[SimilarityMetric]] = None
    ) -> SimilarityScore:
        """
        Analyze similarity between original and regenerated posts using specified metrics.
        
        Args:
            original: Original post
            regenerated: Regenerated post
            neutral: Neutral description used for regeneration
            metrics: List of similarity metrics to use (default: all metrics)
            
        Returns:
            SimilarityScore: Object containing all calculated similarity scores
        """
        if metrics is None:
            metrics = list(SimilarityMetric)
            
        scores = SimilarityScore(
            semantic_similarity=0.0,
            cosine_similarity=0.0
        )
        
        for metric in metrics:
            if metric == SimilarityMetric.SEMANTIC:
                scores.semantic_similarity = self.calculate_semantic_similarity(
                    original, regenerated
                )
                
            elif metric == SimilarityMetric.COSINE:
                scores.cosine_similarity = self.calculate_cosine_similarity(
                    original, regenerated
                )
                
            elif metric == SimilarityMetric.LLM:
                llm_score, explanation = self.get_llm_similarity_judgment(
                    original, regenerated, neutral
                )
                scores.llm_similarity = llm_score
                scores.llm_explanation = explanation
                
        if SimilarityMetric.COMBINED in metrics:
            scores.combined_score = self.calculate_combined_score(scores)
            
        return scores
        
    def analyze_batch(
        self,
        posts: List[Dict[str, str]],
        metrics: Optional[List[SimilarityMetric]] = None
    ) -> List[Dict]:
        """
        Analyze similarity for a batch of posts.
        
        Args:
            posts: List of dictionaries containing original_post, regenerated_post,
                  and neutral_description
            metrics: List of similarity metrics to use
            
        Returns:
            List[Dict]: List of dictionaries containing posts and their similarity scores
        """
        results = []
        total_posts = len(posts)
        
        for idx, post in enumerate(posts, 1):
            logger.info(f"Analyzing post {idx}/{total_posts}")
            
            scores = self.analyze_similarity(
                post['original_post'],
                post['regenerated_post'],
                post['neutral_description'],
                metrics
            )
            
            result = {
                'original_post': post['original_post'],
                'regenerated_post': post['regenerated_post'],
                'neutral_description': post['neutral_description'],
                'semantic_similarity': scores.semantic_similarity,
                'cosine_similarity': scores.cosine_similarity,
                'combined_score': scores.combined_score
            }
            
            if scores.llm_similarity is not None:
                result.update({
                    'llm_similarity': scores.llm_similarity,
                    'llm_explanation': scores.llm_explanation
                })
                
            results.append(result)
            
        return results
