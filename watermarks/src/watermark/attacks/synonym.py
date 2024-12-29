import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from random import choice

class SynonymAttack:
    """Attack that replaces words with synonyms."""
    
    def __init__(self):
        """Initialize NLTK resources if needed."""
        try:
            word_tokenize("test")
        except LookupError:
            nltk.download('punkt')
            nltk.download('averaged_perceptron_tagger')
            nltk.download('wordnet')

    def __call__(self, text: str) -> str:
        """Apply the synonym replacement attack."""
        tokens = word_tokenize(text)
        tagged_tokens = pos_tag(tokens)
        
        new_tokens = []
        for token, tag in tagged_tokens:
            wn_tag = self._get_wordnet_pos(tag)
            if wn_tag:
                # Find synonyms
                synonyms = wn.synsets(token, pos=wn_tag)
                lemmas = [lemma for syn in synonyms 
                         for lemma in syn.lemmas() 
                         if syn.name().split('.')[0] == token]
                if lemmas:
                    # Choose a synonym different from the word itself
                    synonyms = [lemma.name() for lemma in lemmas 
                              if lemma.name() != token]
                    if synonyms:
                        chosen_synonym = choice(synonyms).replace('_', ' ')
                        new_tokens.append(chosen_synonym)
                        continue
            # If no synonym found, keep original word
            new_tokens.append(token)
        
        return ' '.join(new_tokens)

    def _get_wordnet_pos(self, treebank_tag: str):
        """Convert treebank tags to WordNet POS tags."""
        if treebank_tag.startswith('J'):
            return wn.ADJ
        elif treebank_tag.startswith('V'):
            return wn.VERB
        elif treebank_tag.startswith('N'):
            return wn.NOUN
        elif treebank_tag.startswith('R'):
            return wn.ADV
        return None 