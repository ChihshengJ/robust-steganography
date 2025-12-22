import numpy as np
from pathlib import Path
import re
import argparse

def combine_embedding_chunks(embedding_dir):
    """
    Combines all embedding chunks in order, handling arbitrary chunk sizes and missing chunks.
    """
    embedding_dir = Path(embedding_dir)
    
    # Get all chunk files and their ranges
    chunk_info = []
    pattern = re.compile(r'embeddings_(\d+)-(\d+)\.npy')
    
    for chunk_file in embedding_dir.glob("embeddings_*.npy"):
        match = pattern.match(chunk_file.name)
        if not match:
            continue
            
        start_idx = int(match.group(1))
        end_idx = int(match.group(2))
        embeddings = np.load(chunk_file)
        
        # Verify the embeddings shape matches the range in filename
        expected_size = end_idx - start_idx
        if embeddings.shape[0] != expected_size:
            raise ValueError(f"Chunk {chunk_file} contains {embeddings.shape[0]} embeddings but filename indicates {expected_size}")
        
        chunk_info.append({
            'file': chunk_file,
            'embeddings': embeddings,
            'start_idx': start_idx,
            'end_idx': end_idx
        })
    
    if not chunk_info:
        raise ValueError(f"No embedding chunks found in {embedding_dir}")
    
    # Sort chunks by start index
    chunk_info.sort(key=lambda x: x['start_idx'])
    
    # Verify no overlapping ranges and find gaps
    for i in range(len(chunk_info) - 1):
        current_end = chunk_info[i]['end_idx']
        next_start = chunk_info[i + 1]['start_idx']
        
        if current_end > next_start:
            raise ValueError(f"Overlapping chunks found: {chunk_info[i]['file']} and {chunk_info[i + 1]['file']}")
        elif current_end < next_start:
            print(f"\nWarning: Gap in embeddings between paragraphs {current_end}-{next_start}")
    
    # Verify embedding dimension consistency
    embedding_dim = chunk_info[0]['embeddings'].shape[1]
    for chunk in chunk_info[1:]:
        if chunk['embeddings'].shape[1] != embedding_dim:
            raise ValueError(f"Inconsistent embedding dimensions: {chunk['file']} has {chunk['embeddings'].shape[1]} dims, expected {embedding_dim}")
    
    # Combine embeddings
    combined_embeddings = np.concatenate([chunk['embeddings'] for chunk in chunk_info])
    
    # Print summary
    total_start = chunk_info[0]['start_idx']
    total_end = chunk_info[-1]['end_idx']
    print(f"\nProcessed {len(chunk_info)} chunk files:")
    print(f"- Paragraph range: {total_start}-{total_end}")
    print(f"- Total embeddings: {combined_embeddings.shape[0]}")
    print(f"- Embedding dimension: {embedding_dim}")
    
    return combined_embeddings

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Combine embedding chunks')
    parser.add_argument('--chunks-dir', type=str, required=True,
                      help='Directory containing embedding chunks')
    parser.add_argument('--output', type=str, required=True,
                      help='Path to save combined embeddings')
    args = parser.parse_args()
    
    # Process chunks
    combined = combine_embedding_chunks(args.chunks_dir)
    
    # Save combined embeddings
    np.save(args.output, combined)
    print(f"\nSaved combined embeddings to {args.output}")
    print(f"Final shape: {combined.shape}")