from typing import Dict, List, Union
import numpy as np
from rouge_score import rouge_scorer

class RougeEvaluator:
    """Evaluator class for calculating ROUGE metrics between original and regenerated posts."""
    
    def __init__(self, metrics: List[str] = None):
        """
        Initialize the RougeEvaluator.
        
        Args:
            metrics: List of ROUGE metrics to calculate. Defaults to ['rouge1', 'rouge2', 'rougeL']
        """
        self.metrics = metrics or ['rouge1', 'rouge2', 'rougeL']
        self.scorer = rouge_scorer.RougeScorer(self.metrics, use_stemmer=True)
        
    def calculate_rouge_scores(self, original_text: str, generated_text: str) -> Dict[str, Dict[str, float]]:
        """
        Calculate ROUGE scores between original and generated text.
        
        Args:
            original_text: The original text
            generated_text: The generated/regenerated text
            
        Returns:
            Dictionary containing ROUGE scores for each metric
        """
        try:
            scores = self.scorer.score(original_text, generated_text)
            # Convert Score objects to dictionaries with precision, recall, and fmeasure
            return {
                metric: {
                    'precision': scores[metric].precision,
                    'recall': scores[metric].recall,
                    'fmeasure': scores[metric].fmeasure
                }
                for metric in self.metrics
            }
        except Exception as e:
            print(f"Error calculating ROUGE scores: {e}")
            return {metric: {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0} for metric in self.metrics}
    
    def evaluate_batch(self, originals: List[str], generations: List[str]) -> List[Dict[str, Dict[str, float]]]:
        """
        Calculate ROUGE scores for a batch of text pairs.
        
        Args:
            originals: List of original texts
            generations: List of generated texts
            
        Returns:
            List of dictionaries containing ROUGE scores for each pair
        """
        if len(originals) != len(generations):
            raise ValueError("Number of original and generated texts must match")
            
        return [
            self.calculate_rouge_scores(orig, gen)
            for orig, gen in zip(originals, generations)
        ]
    
    @staticmethod
    def aggregate_scores(scores: List[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
        """
        Calculate aggregate statistics for a list of ROUGE scores.
        
        Args:
            scores: List of ROUGE score dictionaries
            
        Returns:
            Dictionary containing mean scores for each metric
        """
        if not scores:
            return {}
            
        # Initialize aggregates
        aggregates: Dict[str, Dict[str, List[float]]] = {}
        
        # Collect all scores
        for score_dict in scores:
            for metric, values in score_dict.items():
                if metric not in aggregates:
                    aggregates[metric] = {'precision': [], 'recall': [], 'fmeasure': []}
                for key, value in values.items():
                    aggregates[metric][key].append(value)
        
        # Calculate means
        return {
            metric: {
                key: float(np.mean(values))
                for key, values in metric_scores.items()
            }
            for metric, metric_scores in aggregates.items()
        }
