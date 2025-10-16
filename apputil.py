from collections import defaultdict


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

        # your code here ...

        return None