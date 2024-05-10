import seaborn as sns
import matplotlib.pylab as plt
import numpy as np
from scipy.spatial.distance import cdist
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from Helpers import find_cutoff, save_plot, save_cutoffs, read_data, plot_embedding_similarities, compute_embeddings_local_pair, compute_embeddings_pair, save_embeddings

# comparison is the data being compared (human_pairs, all_all, etc)
def compare_embeddings(originals, paraphrases, comparison, error_rate, model_name, client, normalize):
    # Compute embeddings
    # embeddings = compute_embeddings_local_pair(originals, paraphrases, model_name)
    embeddings = compute_embeddings_pair(originals, paraphrases, normalize, model_name, client)
    
    # Save embeddings
    save_embeddings(embeddings[0], originals, 'human_ai_originals', normalize, model_name)
    save_embeddings(embeddings[1], paraphrases, 'human_ai_paraphrases', normalize, model_name)
    
    # Make plots
    plot_embedding_similarities(embeddings, normalize, model_name, comparison)

    # Compute similarities
    cosine_matrix = cdist(embeddings[0], embeddings[1], metric='cosine')
    euclidean_matrix = cdist(embeddings[0], embeddings[1], metric='euclidean')

    # find cutoffs
    cutoff_cos = find_cutoff(cosine_matrix, error_rate)
    cutoff_euc = find_cutoff(euclidean_matrix, error_rate)

    # save cutoffs
    save_cutoffs(cutoff_cos, error_rate, 'cosine', comparison, normalize, model_name)
    save_cutoffs(cutoff_euc, error_rate, 'euclidean', comparison, normalize, model_name)

if __name__ == '__main__':

    model_names = [
        # 'paraphrase-MiniLM-L6-v2',
        # 'Salesforce/SFR-Embedding-Mistral',
        # "Alibaba-NLP/gte-Qwen1.5-7B-instruct",
        # "voyage-2",
        # "GritLM/GritLM-7B",
        # "intfloat/e5-mistral-7b-instruct",
        # different code to get embeddings and load model
        # "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-supervised",
    ]

    data = read_data('./text.json')
    originals = data['human_pairs']['originals'] + data['ai_pairs']['originals']
    paraphrases = data['human_pairs']['paraphrases'] + data['ai_pairs']['paraphrases']
    model_name = 'text-embedding-3-large'

    client = OpenAI() # automatically uses OPENAI_API_KEY env var

    compare_embeddings(originals, paraphrases, 'human_and_ai_combined', 0.01, model_name, client, True)
    compare_embeddings(originals, paraphrases, 'human_and_ai_combined', 0.01, model_name, client, False)
    
        
    # except:
    #     print('Error with model: {model_name}'.format(model_name=model_name))
