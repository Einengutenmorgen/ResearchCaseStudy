import pandas as pd
import ast
from pipeline_openai import NeutralDescriptionGenerator
import os
from datetime import datetime
from dotenv import load_dotenv
import time

def run_descriptions(input_csv_path: str):
    # Load environment variables
    load_dotenv()
    
    # Read the CSV file
    print("Reading CSV file...")
    df = pd.read_csv(input_csv_path)
    
    # Initialize the description generator
    generator = NeutralDescriptionGenerator()
    
    # Function to parse the text list
    def parse_text_list(text: str) -> list:
        try:
            return ast.literal_eval(text)
        except:
            return []

    def process_user_posts(row):
        full_text = parse_text_list(row['full_text'])
        
        # Get posts 51-70 for neutral descriptions
        posts_for_neutral = full_text[50:70] if len(full_text) >= 70 else []
        
        if not posts_for_neutral:
            print(f"Skipping user {row['user_id']} - not enough posts")
            return []
        
        # Generate new descriptions
        descriptions = []
        for i, post in enumerate(posts_for_neutral, 1):
            try:
                print(f"Processing post {i}/20 for user {row['user_id']}")
                desc = generator.generate_description(post)
                descriptions.append(desc)
                # Add a small delay to avoid rate limits
                time.sleep(0.5)
            except Exception as e:
                print(f"Error processing post {i} for user {row['user_id']}: {str(e)}")
                descriptions.append("")
        
        return descriptions

    # Process each row
    print("\nGenerating new neutral descriptions...")
    df['new_neutral_descriptions'] = df.apply(process_user_posts, axis=1)
    
    # Create output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.dirname(input_csv_path)
    output_filename = f"updated_descriptions_{timestamp}.csv"
    output_path = os.path.join(output_dir, output_filename)
    
    # Save to new CSV
    print(f"\nSaving results to: {output_path}")
    df.to_csv(output_path, index=False)
    print("Done!")
    return output_path

if __name__ == "__main__":
    # Use the most recent analyzed personas file
    input_csv = "/Users/mogen/Desktop/Research/Results/analyzed_personas_20241122_104657.csv"
    rerun_descriptions(input_csv)
