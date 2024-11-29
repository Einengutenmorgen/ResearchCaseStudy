import pandas as pd
import ast

def read_personas_interactive(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Iterate through each row
    for index, row in df.iterrows():
        print("\n=== Row", index + 1, "===")
        
        # Convert string representation of list back to list for full_text
        try:
            # The full_text is stored as a string representation of a list
            full_text = ast.literal_eval(row['full_text'])
            print("\nFULL TEXT (first 3 tweets):")
            for i, tweet in enumerate(full_text[:3], 1):
                print(f"\nTweet {i}:")
                print(tweet)
            if len(full_text) > 3:
                print(f"\n... and {len(full_text) - 3} more tweets")
        except:
            print("\nFULL TEXT:")
            print(row['full_text'])

        print("\nPERSONA:")
        print(row['persona'])
        
        # Wait for user input
        user_input = input("\nPress Enter to continue or type 'stop' to exit: ")
        if user_input.lower() == 'stop':
            break

if __name__ == "__main__":
    file_path = "/Users/mogen/Desktop/Research/Results/analyzed_personas_20241122_104657.csv"
    read_personas_interactive(file_path)
