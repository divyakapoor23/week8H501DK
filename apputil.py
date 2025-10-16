from collections import defaultdict
import numpy as np


class MarkovText(object):

    def __init__(self, corpus):
        self.corpus = corpus
        self.term_dict = None  # you'll need to build this

    def get_term_dict(self):
        """
        Build a transition dictionary for the Markov chain.
        
        Keys: unique tokens in the corpus
        Values: lists of all tokens that follow each key
        
        We include duplicates in the lists because they represent the frequency
        of transitions, which is important for maintaining the probabilistic
        nature of the Markov chain.
        """
        # Use defaultdict(list) to automatically create empty lists for new keys
        term_dict = defaultdict(list)
        
        # Split the corpus into tokens
        tokens = self.corpus.split()
        
        # Build the transition dictionary
        for i in range(len(tokens) - 1):
            current_token = tokens[i]
            next_token = tokens[i + 1]
            term_dict[current_token].append(next_token)
        
        # Convert to regular dict and store
        self.term_dict = dict(term_dict)
        
        return self.term_dict


    def generate(self, seed_term=None, term_count=15):
        """
        Generate text using the Markov property.
        
        Parameters:
        - seed_term: optional starting word. If None, a random word is chosen.
        - term_count: number of words to generate
        
        Returns:
        - A string of generated text
        """
        import numpy as np
        
        # Build term dictionary if it doesn't exist
        if self.term_dict is None:
            self.get_term_dict()
        
        # Handle seed term
        if seed_term is None:
            # Choose a random starting term from available keys
            current_term = np.random.choice(list(self.term_dict.keys()))
        else:
            # Check if seed term exists in corpus
            if seed_term not in self.term_dict:
                raise ValueError(f"Seed term '{seed_term}' not found in corpus")
            current_term = seed_term
        
        # Initialize the generated text with the starting term
        generated_words = [current_term]
        
        # Generate subsequent words
        for _ in range(term_count - 1):
            # Check if current term has any following words
            if current_term not in self.term_dict or len(self.term_dict[current_term]) == 0:
                # If no following words, break the generation
                break
            
            # Choose next word randomly from the list of possible following words
            next_word = np.random.choice(self.term_dict[current_term])
            generated_words.append(next_word)
            current_term = next_word
        
        # Join the words into a single string
        return ' '.join(generated_words)