import pandas as pd
from collections import defaultdict
import networkx as nx
from typing import Dict, Set, List
import numpy as np

class ConversationExtractor:
    def __init__(self, debug=True):
        self.tweet_replies: Dict[str, Set[str]] = defaultdict(set)
        self.tweet_data: Dict[str, dict] = {}
        self.conversation_graph = nx.DiGraph()
        self.debug = debug
        
    def process_chunk(self, df_chunk: pd.DataFrame) -> None:
        """Process a chunk of the dataset to build conversation mappings."""
        if self.debug:
            print(f"Processing chunk with {len(df_chunk)} rows")
            
        for _, row in df_chunk.iterrows():
            # Handle different types of tweet IDs (int, float, string)
            tweet_id = str(int(row['tweet_id'])) if pd.notnull(row['tweet_id']) else None
            reply_to = str(int(row['reply_to_id'])) if pd.notnull(row['reply_to_id']) else None
            
            if tweet_id is None:
                continue
                
            # Store tweet data
            self.tweet_data[tweet_id] = {
                'text': row['full_text'],
                'user_id': row['original_user_id'],
                'created_at': row['created_at'],
                'screen_name': row['screen_name']
            }
            
            # Add edge to conversation graph if this is a reply
            if reply_to is not None:
                self.tweet_replies[reply_to].add(tweet_id)
                self.conversation_graph.add_edge(reply_to, tweet_id)
                
        if self.debug:
            print(f"Current graph size: {len(self.conversation_graph.nodes())} nodes, "
                  f"{len(self.conversation_graph.edges())} edges")
    
    def process_large_dataset(self, filepath: str, chunk_size: int = 100000) -> None:
        """Process a large dataset in chunks."""
        try:
            chunks = pd.read_csv(filepath, chunksize=chunk_size)
            for i, chunk in enumerate(chunks):
                if self.debug:
                    print(f"Processing chunk {i+1}")
                self.process_chunk(chunk)
                
            if self.debug:
                print("\nDataset Processing Summary:")
                print(f"Total tweets stored: {len(self.tweet_data)}")
                print(f"Total replies mapped: {sum(len(replies) for replies in self.tweet_replies.values())}")
                print(f"Number of root tweets: {len(self.find_root_tweets())}")
                
        except Exception as e:
            print(f"Error processing dataset: {str(e)}")
            raise
    
    def find_root_tweets(self) -> Set[str]:
        """Find all root tweets (tweets that started conversations)."""
        # A root tweet is one that has replies but is not itself a reply
        root_tweets = {node for node in self.conversation_graph.nodes()
                      if self.conversation_graph.in_degree(node) == 0 and
                      self.conversation_graph.out_degree(node) > 0}
        
        if self.debug and not root_tweets:
            print("\nDebugging information:")
            print(f"Total nodes in graph: {len(self.conversation_graph.nodes())}")
            print(f"Total edges in graph: {len(self.conversation_graph.edges())}")
            print("Sample of graph edges:", list(self.conversation_graph.edges())[:5])
            
        return root_tweets

    def get_graph_statistics(self) -> dict:
        """Get detailed statistics about the conversation graph."""
        # Convert to undirected graph for weakly connected components
        undirected_graph = self.conversation_graph.to_undirected()
        
        # Get conversation depths
        depths = []
        for root in self.find_root_tweets():
            try:
                # Get all descendants of this root
                descendants = nx.descendants(self.conversation_graph, root)
                if descendants:
                    # Calculate max depth of this conversation tree
                    depths.append(max(nx.shortest_path_length(self.conversation_graph, root, target)
                                   for target in descendants))
            except nx.NetworkXError:
                continue

        return {
            'total_tweets': len(self.tweet_data),
            'total_edges': len(self.conversation_graph.edges()),
            'total_nodes': len(self.conversation_graph.nodes()),
            'isolated_nodes': len(list(nx.isolates(undirected_graph))),
            'weakly_connected_components': nx.number_connected_components(undirected_graph),
            'root_tweets': len(self.find_root_tweets()),
            'max_conversation_depth': max(depths) if depths else 0,
            'avg_conversation_depth': sum(depths)/len(depths) if depths else 0
        }

    def extract_conversation_thread(self, root_tweet_id: str) -> List[dict]:
        """Extract full conversation thread starting from a root tweet."""
        conversation = []
        
        def traverse_replies(tweet_id: str, depth: int = 0):
            if tweet_id in self.tweet_data:
                tweet_info = self.tweet_data[tweet_id].copy()
                tweet_info['depth'] = depth
                conversation.append(tweet_info)
                
                # Process all replies to this tweet
                for reply_id in self.tweet_replies[tweet_id]:
                    traverse_replies(reply_id, depth + 1)
        
        traverse_replies(root_tweet_id)
        return conversation