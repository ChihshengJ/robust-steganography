"""
Run attacks on inputs and test if they succeed.
"""
import json
import base64
from attacks import synonym_attack
from steg import decode
from Helpers import detect, get_limit, to_tokens, get_ids, generate_n_grams_with_counts, count_maintained_n_grams_with_frequency

def load_examples():
    """Loads examples from the data.json file."""
    with open('data.json', 'r') as f:
        data = json.load(f)
    # convert keys to bytes
    for example in data:
        example['keys'] = [base64.b64decode(key) for key in example['keys']]
    return data

def log_synonym_attack(original, modified, keys, h, m, c, original_n_grams, n_grams_common, survived):
    with open('./attacks/synonym_attack.log', "a") as myfile:
      myfile.write('original: {original}\n'.format(original=original))
      myfile.write('modified: {modified}\n'.format(modified=modified))
      myfile.write('keys: {keys}\n'.format(keys=keys))
      myfile.write('h: {h}\n'.format(h=h))
      myfile.write('m: {m}\n'.format(m=m))
      myfile.write('c: {c}\n'.format(c=c))
      myfile.write('original_n_grams: {original_n_grams}\n'.format(original_n_grams=original_n_grams))
      myfile.write('n_grams_common: {n_grams_common}\n'.format(n_grams_common=n_grams_common))
      myfile.write('n-gram % maintained: {percent}\n'.format(percent=n_grams_common/original_n_grams))
      myfile.write('survived: {survived}\n'.format(survived=survived))
      myfile.write('-----------------------------------\n')
      

def test_synonym_attack(examples):
    """Tests the synonym_attack function."""
    survived = []
    for example in examples:
        # get covertext
        original = example['ct']
        # apply attack
        modified = synonym_attack(original)
        print(original)
        print('-----------------------------------')
        print(modified)
        # attempt to recover the message
        keys = example['keys']
        h = example['h']
        c = example['c']
        recovered_counters, decode_tokens = decode(keys, h, modified, None, c)
        m_prime = [1 if detect(get_limit(None), x) else 0 for x in recovered_counters]
        # also want to measure c-grams that stay intact
        original_n_grams = sum(generate_n_grams_with_counts(get_ids(to_tokens(original)), c).values())
        n_grams_common = count_maintained_n_grams_with_frequency(get_ids(to_tokens(original)), get_ids(to_tokens(modified)), c)
        log_synonym_attack(original, modified, keys, h, example['m'], c, original_n_grams, n_grams_common, m_prime)
        if m_prime == example['m']:
            survived.append((original, modified))
    print('Survided:', survived)
    print('Survived attacks:', len(survived))
    return survived



examples = load_examples()

# 38/99 survived
test_synonym_attack(examples)