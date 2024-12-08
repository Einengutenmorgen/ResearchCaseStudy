import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import pandas as pd
from datetime import datetime
import openai
from dotenv import load_dotenv
from rouge_score import rouge_scorer
from user_post_collector import UserPostCollector
from persona_manager import PersonaManager
from tqdm import tqdm


# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConversationFilter:
    def __init__(self, min_messages: int = 4):
        """
        Initialize the ConversationFilter.
        
        Args:
            min_messages: Minimum number of messages required for a valid conversation
        """
        self.min_messages = min_messages
        
    def filter_conversations(self, conv_directory: str) -> List[Dict]:
        """
        Filter conversations based on length and prepare them for processing.
        
        Args:
            conv_directory: Directory containing conversation JSON files
            
        Returns:
            List of filtered conversations with held out messages
        """
        filtered_convs = []
        conv_dir = Path(conv_directory) / "batch_chunk_0000"
        logger.info(f"Looking for conversations in {conv_dir}")
        
        if not conv_dir.exists():
            logger.error(f"Conversation directory {conv_dir} does not exist")
            return []

        try:
            # Count total files first for accurate progress bar
            total_files = sum(1 for _ in conv_dir.glob("*.json"))
            logger.info(f"Found {total_files} conversation files to process")

            # Use iterator for memory efficiency
            json_files = conv_dir.glob("*.json")
            
            # Create progress bar with total count
            for conv_file in tqdm(json_files, total=total_files, desc="Processing conversations", 
                                unit="files", ncols=80):
                try:
                    # Add timeout for file reading
                    with open(conv_file) as f:
                        conv = json.load(f)
                    
                    if len(conv["messages"]) >= self.min_messages:
                        # Extract participants from messages
                        participants = set()
                        for msg in conv["messages"]:
                            participants.add(msg["screen_name"])
                            if msg["reply_to_user"] and msg["reply_to_user"] != "null":
                                participants.add(str(msg["reply_to_user"]))
                        
                        # Separate last message as ground truth
                        processed_conv = {
                            "conversation_id": conv_file.stem,
                            "history": conv["messages"][:-1],
                            "ground_truth": conv["messages"][-1],
                            "participants": list(participants),
                            "metadata": conv["metadata"]
                        }
                        filtered_convs.append(processed_conv)
                        
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error in {conv_file}: {str(e)}")
                    continue
                except Exception as e:
                    logger.error(f"Error processing {conv_file}: {str(e)}")
                    continue
            
        except Exception as e:
            logger.error(f"Fatal error during conversation filtering: {str(e)}")
            return []
        finally:
            # Always log the results, even if there was an error
            logger.info(f"Filtered {len(filtered_convs)} conversations with {self.min_messages}+ messages")
        
        return filtered_convs

class PromptGenerator:
    def __init__(self):
        """Initialize the PromptGenerator."""
        self.system_message = """You are participating in a Twitter conversation. 
Given the conversation history and participant personas, generate the next message 
in the thread that maintains the conversation's style, context, and flow. Your response 
should be natural and contextually appropriate, reflecting the communication patterns 
of the participants."""
        
    def create_prompt(self, conversation: Dict, personas: Dict[str, Dict]) -> Dict:
        """
        Create a formatted prompt for the model.
        
        Args:
            conversation: Processed conversation dictionary
            personas: Dictionary mapping user IDs to their personas
            
        Returns:
            Formatted prompt for the model
        """
        messages = [{"role": "system", "content": self.system_message}]
        
        # Add conversation context with personas
        context = self._build_context(conversation, personas)
        messages.append({"role": "user", "content": context})
        
        # Add conversation history
        history = self._format_conversation_history(conversation["history"])
        messages.append({"role": "user", "content": history})
        
        return messages
        
    def _build_context(self, conversation: Dict, personas: Dict[str, Dict]) -> str:
        """Build context description including participant personas."""
        context = [f"This is a Twitter conversation with {len(conversation['participants'])} participants."]
        
        # Add duration
        duration_hours = conversation['metadata']['time_span']['duration_seconds']/3600
        context.append(f"The conversation has been ongoing for {duration_hours:.1f} hours.")
        
        # Add participant personas
        context.append("\nParticipant Information:")
        for participant in conversation['participants']:
            if participant in personas:
                context.append(f"\nUser {participant}:")
                context.append(personas[participant]['analysis'])
        
        context.append("\nYour task is to generate the next message in this thread, maintaining consistency with the participants' personas and conversation style.")
        
        return "\n".join(context)
        
    def _format_conversation_history(self, history: List[Dict]) -> str:
        """Format conversation history into a readable string."""
        formatted_msgs = []
        for msg in history:
            formatted_msgs.append(
                f"@{msg['screen_name']}: {msg['full_text']}"
            )
        return "\n".join(formatted_msgs)

class MessageGenerator:
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        """
        Initialize the MessageGenerator.
        
        Args:
            api_key: OpenAI API key
            model: Model to use for generation
        """
        self.model = model
        self.client = openai.OpenAI(api_key=api_key)
        
    def generate_message(self, prompt: List[Dict]) -> str:
        """
        Generate next message using the OpenAI API.
        
        Args:
            prompt: Formatted prompt messages
            
        Returns:
            Generated message
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=prompt,
                temperature=0.7,
                max_tokens=150,
                n=1
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating message: {str(e)}")
            raise

class ResponseEvaluator:
    def __init__(self):
        """Initialize the ResponseEvaluator."""
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
    def evaluate_response(self, generated: str, ground_truth: str) -> Dict:
        """
        Evaluate the generated response against ground truth.
        
        Args:
            generated: Generated message
            ground_truth: Actual next message
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Calculate ROUGE scores
        scores = self.scorer.score(ground_truth, generated)
        
        return {
            "rouge1_f": scores['rouge1'].fmeasure,
            "rouge2_f": scores['rouge2'].fmeasure,
            "rougeL_f": scores['rougeL'].fmeasure,
            "generated_length": len(generated.split()),
            "ground_truth_length": len(ground_truth.split())
        }

class ResultsManager:
    def __init__(self, output_dir: str):
        """
        Initialize the ResultsManager.
        
        Args:
            output_dir: Directory to store results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def save_results(self, results: List[Dict]) -> str:
        """
        Save evaluation results.
        
        Args:
            results: List of result dictionaries
            
        Returns:
            Path to saved results
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"prediction_results_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Saved results to {output_file}")
        return str(output_file)

def main(conv_directory: str, data_file: str, output_dir: str, api_key: str):
    """
    Run the conversation prediction pipeline.
    
    Args:
        conv_directory: Directory containing conversation files
        data_file: Path to the original CSV file containing all tweets
        output_dir: Directory to store results
        api_key: OpenAI API key
    """
    # Initialize components
    conv_filter = ConversationFilter(min_messages=4)
    prompt_gen = PromptGenerator()
    msg_gen = MessageGenerator(api_key)
    evaluator = ResponseEvaluator()
    results_mgr = ResultsManager(output_dir)
    
    # Process conversations
    filtered_convs = conv_filter.filter_conversations(conv_directory)
    logger.info(f"Filtered {len(filtered_convs)} conversations")
    
    # Collect all unique participants
    all_participants = set()
    for conv in filtered_convs:
        all_participants.update(conv["participants"])
    logger.info(f"Found {len(all_participants)} unique participants")
    
    # Collect user posts and generate personas
    post_collector = UserPostCollector(data_file)
    user_posts = post_collector.collect_user_posts(all_participants)
    
    persona_mgr = PersonaManager(api_key, cache_dir=Path(output_dir) / "persona_cache")
    personas = {}
    for user_id, posts in user_posts.items():
        personas[user_id] = persona_mgr.generate_persona(user_id, posts)
    
    results = []
    for conv in filtered_convs:
        try:
            # Get relevant personas for this conversation
            conv_personas = {
                pid: personas.get(pid, {"analysis": "No persona available"})
                for pid in conv["participants"]
            }
            
            # Generate prompt
            prompt = prompt_gen.create_prompt(conv, conv_personas)
            
            # Generate response
            generated_msg = msg_gen.generate_message(prompt)
            
            # Evaluate
            evaluation = evaluator.evaluate_response(
                generated_msg,
                conv["ground_truth"]["full_text"]
            )
            
            # Store results
            result = {
                "conversation_id": conv["conversation_id"],
                "history": conv["history"],
                "generated_message": generated_msg,
                "ground_truth": conv["ground_truth"],
                "metrics": evaluation,
                "personas_used": conv_personas
            }
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error processing conversation {conv['conversation_id']}: {str(e)}")
            continue
            
    # Save results
    results_file = results_mgr.save_results(results)
    logger.info(f"Pipeline completed. Results saved to {results_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run conversation prediction pipeline')
    parser.add_argument('conv_directory', help='Directory containing conversation JSON files')
    parser.add_argument('data_file', help='Path to the original CSV file containing all tweets')
    parser.add_argument('output_dir', help='Directory to store results')
    parser.add_argument('--api-key', help='OpenAI API key (optional, defaults to OPENAI_API_KEY env variable)',
                      default=os.getenv('OPENAI_API_KEY'))
    
    args = parser.parse_args()
    
    if not args.api_key:
        raise ValueError("OpenAI API key must be provided either via --api-key argument or OPENAI_API_KEY environment variable")
    
    main(args.conv_directory, args.data_file, args.output_dir, args.api_key)
