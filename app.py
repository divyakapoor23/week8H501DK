import streamlit as st
import requests
import re
import pandas as pd
from apputil import MarkovText
from collections import defaultdict
import numpy as np
import random


class EnhancedMarkovText(object):

    def __init__(self, corpus, k=1):
        """
        Initialize the enhanced Markov text generator.
        
        Parameters:
        - corpus: text corpus for training
        - k: state window size (number of words to use as state)
        """
        self.corpus = corpus
        self.k = k
        self.term_dict = None
    
    def get_term_dict(self):
        """
        Build a transition dictionary with k-word states.
        
        For k=1: {"word": ["next1", "next2", ...]}
        For k=2: {("word1", "word2"): ["next1", "next2", ...]}
        For k=3: {("word1", "word2", "word3"): ["next1", "next2", ...]}
        """
        term_dict = defaultdict(list)
        tokens = self.corpus.split()
        
        # Build k-word states
        for i in range(len(tokens) - self.k):
            if self.k == 1:
                # Single word state (original implementation)
                current_state = tokens[i]
            else:
                # Multi-word state (tuple)
                current_state = tuple(tokens[i:i+self.k])
            
            next_word = tokens[i + self.k]
            term_dict[current_state].append(next_word)
        
        self.term_dict = dict(term_dict)
        return self.term_dict
    
    def generate(self, seed_term=None, term_count=15):
        """
        Generate text using k-word Markov states.
        
        Parameters:
        - seed_term: starting word(s). Can be a string (for k=1) or tuple (for k>1)
        - term_count: number of words to generate
        """
        if self.term_dict is None:
            self.get_term_dict()
        
        # Handle seed term based on k value
        if seed_term is None:
            # Choose random starting state - use random.choice for complex objects
            current_state = random.choice(list(self.term_dict.keys()))
        else:
            # Validate and set seed term
            if self.k == 1:
                if seed_term not in self.term_dict:
                    raise ValueError(f"Seed term '{seed_term}' not found in corpus")
                current_state = seed_term
            else:
                # For k>1, seed_term should be a tuple or we'll try to find a matching state
                if isinstance(seed_term, str):
                    # Find a state that starts with this word
                    matching_states = [state for state in self.term_dict.keys() 
                                     if state[0] == seed_term]
                    if not matching_states:
                        raise ValueError(f"No states found starting with '{seed_term}'")
                    current_state = random.choice(matching_states)
                else:
                    if seed_term not in self.term_dict:
                        raise ValueError(f"Seed state '{seed_term}' not found in corpus")
                    current_state = seed_term
        
        # Initialize generated words
        if self.k == 1:
            generated_words = [current_state]
        else:
            generated_words = list(current_state)
        
        # Generate subsequent words
        for _ in range(term_count - self.k):
            if current_state not in self.term_dict or len(self.term_dict[current_state]) == 0:
                break
            
            # Choose next word - use numpy for lists
            next_word = np.random.choice(self.term_dict[current_state])
            generated_words.append(next_word)
            
            # Update current state for next iteration
            if self.k == 1:
                current_state = next_word
            else:
                # Slide the window: remove first word, add new word
                current_state = current_state[1:] + (next_word,)
        
        return ' '.join(generated_words)

# Page configuration
st.set_page_config(
    page_title="Markov Text Generator",
    page_icon="📝",
    layout="wide"
)

st.write('''
# Week 8: Natural Language Processing - Markov Text Generator

This interactive app demonstrates **Markov Chain text generation** using inspirational quotes. 
Explore the transition dictionary (Exercise 1), basic text generation (Exercise 2), and advanced k-word states (Bonus Exercise).

🎯 **Features**: Dictionary exploration, text generation, coherence comparison, and advanced multi-word context analysis.
''')

# Sidebar for navigation
st.sidebar.title("🎛️ Controls")
exercise = st.sidebar.radio(
    "Choose Exercise:",
    ["Exercise 1: Transition Dictionary", "Exercise 2: Text Generation", "Bonus: K-Word States", "All Exercises"]
)

# Load and prepare the corpus
@st.cache_data
def load_corpus():
    """Load and preprocess the inspirational quotes corpus"""
    try:
        url = 'https://raw.githubusercontent.com/leontoddjohnson/datasets/main/text/inspiration_quotes.txt'
        content = requests.get(url)
        quotes_raw = content.text
        
        # Clean the data
        quotes = quotes_raw.replace('\n', ' ')
        
        # Try different approaches to split the quotes
        try:
            # First try with the special quote characters
            quotes = re.split(r'[""]', quotes)  # Use raw string for regex
        except:
            # Fallback: split on regular quotes
            quotes = re.split(r'["""]', quotes)
        
        # Filter out empty strings and get every other element
        quotes = [q.strip() for q in quotes if q.strip()]
        if len(quotes) > 1:
            quotes = quotes[1::2]  # Extract quotes
        
        # If we still don't have quotes, try a different approach
        if not quotes:
            # Split on periods and exclamation marks to get sentences
            quotes = re.split(r'[.!]\s+', quotes_raw)
            quotes = [q.strip() for q in quotes if len(q.strip()) > 20]  # Keep sentences with substance
        
        # Create corpus
        corpus = ' '.join(quotes)
        corpus = re.sub(r"\s+", " ", corpus)  # Remove multiple whitespaces
        corpus = corpus.strip()
        
        # Validate corpus
        if len(corpus.split()) < 50:  # Need at least 50 words
            raise ValueError("Corpus too small for meaningful analysis")
        
        return corpus, quotes[:10]  # Return corpus and sample quotes
        
    except Exception as e:
        st.error(f"Error loading corpus: {e}")
        
        # Fallback: use a simple demo corpus
        demo_corpus = """
        Life is what happens when you're busy making other plans. 
        The only way to do great work is to love what you do. 
        Innovation distinguishes between a leader and a follower. 
        Stay hungry, stay foolish. 
        The future belongs to those who believe in the beauty of their dreams.
        Happiness is not something ready made. It comes from your own actions.
        The best time to plant a tree was 20 years ago. The second best time is now.
        """
        demo_quotes = [
            "Life is what happens when you're busy making other plans.",
            "The only way to do great work is to love what you do.",
            "Innovation distinguishes between a leader and a follower."
        ]
        return demo_corpus.strip(), demo_quotes

# Load data
corpus, sample_quotes = load_corpus()

if corpus is None or len(corpus.split()) < 10:
    st.error("Failed to load a valid corpus. Please check your internet connection or try again later.")
    st.stop()

# Display corpus info
st.sidebar.write(f"📊 Corpus loaded: {len(corpus.split())} words")

# Create MarkovText instance
markov_gen = MarkovText(corpus)

# Exercise 1: Transition Dictionary
if exercise in ["Exercise 1: Transition Dictionary", "All Exercises"]:
    st.header("📚 Exercise 1: Transition Dictionary Explorer")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📖 Sample Quotes")
        if sample_quotes:
            for i, quote in enumerate(sample_quotes[:5], 1):
                st.write(f"{i}. *\"{quote}\"*")
    
    with col2:
        st.subheader("🔍 Dictionary Statistics")
        
        # Build term dictionary
        with st.spinner("Building transition dictionary..."):
            term_dict = markov_gen.get_term_dict()
        
        # Display statistics
        st.metric("Total Unique Words", len(term_dict))
        st.metric("Total Word Transitions", sum(len(transitions) for transitions in term_dict.values()))
        
        # Most connected words
        most_connected = sorted(term_dict.items(), key=lambda x: len(x[1]), reverse=True)[:5]
        st.write("**Most Connected Words:**")
        for word, transitions in most_connected:
            st.write(f"• **{word}**: {len(transitions)} transitions")
    
    # Word explorer
    st.subheader("🔎 Word Transition Explorer")
    
    # Search for specific word
    search_word = st.text_input("Search for a word to see its transitions:", placeholder="e.g., life, happiness, love")
    
    if search_word:
        if search_word.lower() in term_dict:
            transitions = term_dict[search_word.lower()]
            st.success(f"Found **{len(transitions)}** transitions for '{search_word}':")
            
            # Show unique transitions with counts
            from collections import Counter
            transition_counts = Counter(transitions)
            
            # Create DataFrame for better display
            df = pd.DataFrame([
                {"Next Word": word, "Frequency": count, "Probability": f"{count/len(transitions):.2%}"}
                for word, count in transition_counts.most_common(10)
            ])
            st.dataframe(df, width=600)
            
            # Show all transitions (truncated for display)
            with st.expander("See all transitions"):
                st.write(" → ".join(transitions[:50]) + ("..." if len(transitions) > 50 else ""))
        else:
            st.warning(f"Word '{search_word}' not found in corpus.")
    
    # Random word exploration
    if st.button("🎲 Explore Random Word"):
        if term_dict and len(term_dict) > 0:
            import random
            random_word = random.choice(list(term_dict.keys()))
            transitions = term_dict[random_word]
            st.info(f"**Random word: '{random_word}'**")
            st.write(f"Transitions: {transitions[:10]}{'...' if len(transitions) > 10 else ''}")
        else:
            st.warning("No words available in the dictionary. Please check the corpus loading.")

# Exercise 2: Text Generation
if exercise in ["Exercise 2: Text Generation", "All Exercises"]:
    st.header("✨ Exercise 2: Markov Text Generator")
    
    # Generation controls
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        word_count = st.slider("Number of words to generate:", min_value=5, max_value=50, value=20)
    
    with col2:
        seed_option = st.radio("Starting word:", ["Random", "Custom"])
    
    with col3:
        if seed_option == "Custom":
            seed_word = st.text_input("Enter seed word:", placeholder="e.g., life, happiness")
        else:
            seed_word = None
    
    # Generate button
    if st.button("🚀 Generate Inspirational Text", type="primary"):
        try:
            if seed_option == "Custom" and seed_word:
                generated_text = markov_gen.generate(seed_term=seed_word.lower(), term_count=word_count)
                st.success(f"**Generated text starting with '{seed_word}':**")
            else:
                generated_text = markov_gen.generate(term_count=word_count)
                st.success("**Generated text with random start:**")
            
            # Display generated text in a nice format
            st.markdown(f"> *{generated_text}*")
            
            # Analysis
            words = generated_text.split()
            st.info(f"Generated {len(words)} words. Click again for different results!")
            
        except ValueError as e:
            st.error(f"Error: {e}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")
    
    # Multiple generations for comparison
    st.subheader("🔄 Compare Multiple Generations")
    
    if st.button("Generate 3 Different Texts"):
        st.write("**Comparing 3 different generations:**")
        
        for i in range(3):
            try:
                if seed_option == "Custom" and seed_word:
                    text = markov_gen.generate(seed_term=seed_word.lower(), term_count=word_count)
                else:
                    text = markov_gen.generate(term_count=word_count)
                
                st.write(f"**{i+1}.** *{text}*")
            except Exception as e:
                st.write(f"**{i+1}.** Error: {e}")

# Bonus Exercise: K-Word State Windows
if exercise in ["Bonus: K-Word States", "All Exercises"]:
    st.header("🎯 Bonus Exercise: K-Word State Windows")
    
    st.markdown("""
    **Advanced Markov Chains**: Instead of using single words as states, we can use sequences of k words.
    This creates more coherent text by considering more context!
    """)
    
    # K-value selection
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        k_value = st.selectbox("State window size (k):", [1, 2, 3], index=1)
    
    with col2:
        bonus_word_count = st.slider("Words to generate:", min_value=10, max_value=40, value=25, key="bonus_words")
    
    with col3:
        st.markdown(f"""
        **k={k_value}**: Using {k_value}-word state{'s' if k_value > 1 else ''}
        
        - k=1: Traditional single-word states
        - k=2: Two-word context (more coherent)
        - k=3: Three-word context (most coherent)
        """)
    
    # Create enhanced generator
    enhanced_gen = EnhancedMarkovText(corpus, k=k_value)
    
    # Build and display dictionary stats
    with st.spinner(f"Building {k_value}-word state dictionary..."):
        enhanced_dict = enhanced_gen.get_term_dict()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(f"Total {k_value}-word States", len(enhanced_dict))
        
        # Show sample states
        st.subheader("🔍 Sample States")
        sample_states = list(enhanced_dict.keys())[:5]
        for i, state in enumerate(sample_states, 1):
            if k_value == 1:
                st.write(f"{i}. **'{state}'** → {enhanced_dict[state][:3]}...")
            else:
                state_str = " ".join(state)
                st.write(f"{i}. **'{state_str}'** → {enhanced_dict[state][:3]}...")
    
    with col2:
        st.metric("Avg Transitions per State", f"{sum(len(v) for v in enhanced_dict.values()) / len(enhanced_dict):.1f}")
        
        # Most connected states
        most_connected = sorted(enhanced_dict.items(), key=lambda x: len(x[1]), reverse=True)[:3]
        st.subheader("🔗 Most Connected States")
        for state, transitions in most_connected:
            if k_value == 1:
                st.write(f"**'{state}'**: {len(transitions)} transitions")
            else:
                state_str = " ".join(state)
                st.write(f"**'{state_str}'**: {len(transitions)} transitions")
    
    # Text generation
    st.subheader("✨ Enhanced Text Generation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        seed_type = st.radio("Starting state:", ["Random", "Custom"], key="bonus_seed")
    
    with col2:
        if seed_type == "Custom":
            if k_value == 1:
                bonus_seed = st.text_input("Enter starting word:", key="bonus_seed_word")
            else:
                st.write("Enter a word to find matching states:")
                bonus_seed = st.text_input("Starting word:", key="bonus_seed_multi")
        else:
            bonus_seed = None
    
    # Generate button
    if st.button("🚀 Generate Enhanced Text", type="primary", key="bonus_generate"):
        try:
            if seed_type == "Custom" and bonus_seed:
                generated_text = enhanced_gen.generate(seed_term=bonus_seed.lower(), term_count=bonus_word_count)
                st.success(f"**Generated text with k={k_value} starting with '{bonus_seed}':**")
            else:
                generated_text = enhanced_gen.generate(term_count=bonus_word_count)
                st.success(f"**Generated text with k={k_value} (random start):**")
            
            # Display generated text
            st.markdown(f"> *{generated_text}*")
            
            # Analysis
            words = generated_text.split()
            st.info(f"Generated {len(words)} words using {k_value}-word context window!")
            
        except ValueError as e:
            st.error(f"Error: {e}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")
    
    # Comparison across different k values
    st.subheader("📊 Compare K-Values")
    
    if st.button("🔍 Compare k=1, k=2, k=3", key="compare_k"):
        st.write("**Comparing text coherence across different k values:**")
        
        for k in [1, 2, 3]:
            try:
                temp_gen = EnhancedMarkovText(corpus, k=k)
                temp_gen.get_term_dict()
                
                if seed_type == "Custom" and bonus_seed:
                    text = temp_gen.generate(seed_term=bonus_seed.lower(), term_count=20)
                else:
                    text = temp_gen.generate(term_count=20)
                
                st.write(f"**k={k}**: *{text}*")
                
            except Exception as e:
                st.write(f"**k={k}**: Error - {e}")
        
        st.markdown("""
        **📈 Observations:**
        - **k=1**: More random, creative but less coherent
        - **k=2**: Better phrase structure and flow
        - **k=3**: Most coherent and grammatically correct
        - **Higher k**: Requires larger corpus to avoid repetition
        """)

# Footer with information
st.markdown("---")
st.markdown("""
### 🧠 How It Works:
- **Exercise 1**: Build transition dictionaries mapping words to their following words
- **Exercise 2**: Generate text using single-word Markov states and random selection
- **Bonus Exercise**: Advanced k-word states for improved text coherence
- **Markov Property**: Each state depends only on the previous k states (memoryless)
- **Monte Carlo Simulation**: Iterative sampling from probability distributions

### 🎯 Key Insights:
- **k=1**: Fast, creative, but chaotic text generation
- **k=2**: Balanced coherence and creativity  
- **k=3**: Most coherent but potentially repetitive
- **Trade-off**: Higher k values need larger training corpora

*Built with Streamlit • Data: Inspirational Quotes • Advanced NLP Concepts*
""")

