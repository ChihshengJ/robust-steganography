import nltk
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('punkt')
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from random import choice

def get_wordnet_pos(treebank_tag):
    """Converts treebank tags to WordNet tags."""
    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        return None

def synonym_attack(text):
    """Replaces words in the input text with their synonyms."""
    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)
    
    new_tokens = []
    for token, tag in tagged_tokens:
        wn_tag = get_wordnet_pos(tag)
        if wn_tag:
            # Find synonyms
            synonyms = wn.synsets(token, pos=wn_tag)
            lemmas = [lemma for syn in synonyms for lemma in syn.lemmas() if syn.name().split('.')[0] == token]
            if lemmas:
                # Choose a synonym different from the word itself, if possible
                synonyms = [lemma.name() for lemma in lemmas if lemma.name() != token]
                if synonyms:
                    chosen_synonym = choice(synonyms).replace('_', ' ')
                    new_tokens.append(chosen_synonym)
                    continue
        # If no synonym was found or applicable, keep the original word
        new_tokens.append(token)
    
    return ' '.join(new_tokens)


