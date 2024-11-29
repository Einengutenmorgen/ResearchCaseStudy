import pandas as pd
import ast
from typing import List, Tuple
import textwrap
from colorama import init, Fore, Style
init()  # Initialize colorama

class DescriptionComparer:
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        self.current_index = 0
        self.filter_text = ""
        
    def parse_text_list(self, text: str) -> List[str]:
        """Parse the text from string representation of list to actual list."""
        try:
            return ast.literal_eval(text)
        except:
            return [text] if isinstance(text, str) else []
            
    def format_text(self, text: str, width: int = 80) -> str:
        """Format text with word wrapping."""
        return "\n".join(textwrap.wrap(text, width=width))
    
    def display_comparison(self, index: int):
        """Display the comparison for a specific index."""
        if index < 0 or index >= len(self.df):
            print(f"{Fore.RED}Invalid index{Style.RESET_ALL}")
            return
        
        row = self.df.iloc[index]
        
        # Parse the texts
        original_posts = self.parse_text_list(row['full_text'])
        neutral_descriptions = self.parse_text_list(row['neutral_descriptions'])
        
        print(f"\n{Fore.CYAN}=== Comparison {index + 1}/{len(self.df)} ==={Style.RESET_ALL}")
        print(f"{Fore.YELLOW}User ID: {row['user_id']}{Style.RESET_ALL}\n")
        
        # Get posts 51-70 for comparison with neutral descriptions
        posts_for_neutral = original_posts[50:70] if len(original_posts) > 50 else []
        
        # Display comparisons side by side
        max_comparisons = min(len(posts_for_neutral), len(neutral_descriptions))
        for i in range(max_comparisons):
            print(f"{Fore.GREEN}Original Post {i + 51}:{Style.RESET_ALL}")  # Adjusted post number
            print(self.format_text(posts_for_neutral[i]))
            print(f"\n{Fore.BLUE}Neutral Description {i + 1}:{Style.RESET_ALL}")
            print(self.format_text(neutral_descriptions[i]))
            print("\n" + "="*80 + "\n")
    
    def search_posts(self, search_text: str) -> List[int]:
        """Search for posts containing specific text."""
        indices = []
        for idx, row in self.df.iterrows():
            posts = self.parse_text_list(row['full_text'])
            descriptions = self.parse_text_list(row['neutral_descriptions'])
            
            if any(search_text.lower() in post.lower() for post in posts) or \
               any(search_text.lower() in desc.lower() for desc in descriptions):
                indices.append(idx)
        return indices
    
    def run(self):
        """Run the interactive comparison tool."""
        while True:
            self.display_comparison(self.current_index)
            
            command = input(f"\n{Fore.YELLOW}Commands:{Style.RESET_ALL}\n"
                          "n: next | p: previous | s: search | f: first | l: last | q: quit\n"
                          "Enter command: ").lower().strip()
            
            if command == 'q':
                break
            elif command == 'n':
                self.current_index = min(self.current_index + 1, len(self.df) - 1)
            elif command == 'p':
                self.current_index = max(self.current_index - 1, 0)
            elif command == 'f':
                self.current_index = 0
            elif command == 'l':
                self.current_index = len(self.df) - 1
            elif command == 's':
                search_text = input("Enter search text: ")
                matches = self.search_posts(search_text)
                if matches:
                    print(f"\nFound {len(matches)} matches!")
                    self.current_index = matches[0]
                else:
                    print("\nNo matches found.")
            else:
                print(f"{Fore.RED}Invalid command{Style.RESET_ALL}")

if __name__ == "__main__":
    csv_path = "/Users/mogen/Desktop/Research/Results/analyzed_personas_20241122_104657.csv"
    comparer = DescriptionComparer(csv_path)
    comparer.run()
