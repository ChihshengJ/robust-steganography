from Helpers import create_all_text, save_embeddings, plot_embedding_similarities, compute_embeddings_local_pair

if __name__ == '__main__':
    all = create_all_text()
    embeddings = compute_embeddings_local_pair(all, all, 'paraphrase-MiniLM-L6-v2')
    save_embeddings(embeddings[0], all, 'embeddings_1', 'paraphrase-MiniLM-L6-v2')
    save_embeddings(embeddings[1], all, 'embeddings_2', 'paraphrase-MiniLM-L6-v2')
    plot_embedding_similarities(embeddings, 'paraphrase-MiniLM-L6-v2', 'all_all')
