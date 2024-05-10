from openai import OpenAI
from Helpers import read_data, compute_embeddings_pair, plot_embedding_similarities, create_all_text

if __name__ == "__main__":
    # Set your OpenAI API key here
    client = OpenAI() # automatically uses OPENAI_API_KEY env var
    all = create_all_text()
    embeddings = compute_embeddings_pair(all, all, "text-embedding-3-large", client)
    # embeddings = [
    #     read_data('./embeddings/text-embedding-3-large/embeddings_1.txt')['embedding'],
    #     read_data('./embeddings/text-embedding-3-large/embeddings_2.txt')['embedding']
    # ]
    # save embeddings
    # save_embeddings(embeddings[0], all, 'embeddings_1', "text-embedding-3-large")
    # save_embeddings(embeddings[1], all, 'embeddings_2', "text-embedding-3-large")
    
    plot_embedding_similarities(embeddings, "text-embedding-3-large", "all_all")

