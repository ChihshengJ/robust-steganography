import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
from robust_steganography.utils.get_embedding import get_embeddings_in_batch
import argparse

def process_paragraphs_in_chunks(json_path, output_dir, chunk_size=100, specific_ranges=None):
    """
    Process paragraphs from JSON file in chunks and save their embeddings.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load paragraphs from JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        paragraphs = [p['text'] for p in data['paragraphs']]
    
    client = OpenAI()
    failed_ranges = []
    
    # Determine ranges to process
    if specific_ranges:
        ranges_to_process = specific_ranges
    else:
        total_paragraphs = len(paragraphs)
        ranges_to_process = [
            (i, min(i + chunk_size, total_paragraphs))
            for i in range(0, total_paragraphs, chunk_size)
        ]
    
    # Process chunks
    for start_idx, end_idx in tqdm(ranges_to_process, desc="Processing chunks"):
        output_path = output_dir / f'embeddings_{start_idx:06d}-{end_idx:06d}.npy'
        
        # Skip if chunk already exists
        if output_path.exists():
            print(f"Range {start_idx}-{end_idx} already exists, skipping...")
            continue
        
        # Get chunk of paragraphs
        chunk_paragraphs = paragraphs[start_idx:end_idx]
        
        try:
            # Get embeddings
            embeddings = get_embeddings_in_batch(client, chunk_paragraphs)
            embeddings_array = np.array(embeddings)
            
            # Verify we got the expected number of embeddings
            expected_size = end_idx - start_idx
            if embeddings_array.shape[0] != expected_size:
                raise ValueError(f"Got {embeddings_array.shape[0]} embeddings for {expected_size} paragraphs")
            
            # Save chunk
            np.save(output_path, embeddings_array)
            print(f"Saved range {start_idx}-{end_idx} with shape {embeddings_array.shape}")
            
        except Exception as e:
            print(f"Error processing range {start_idx}-{end_idx}: {str(e)}")
            failed_ranges.append((start_idx, end_idx))
            continue
    
    return failed_ranges

def main():
    parser = argparse.ArgumentParser(description='Process paragraphs and generate embeddings')
    parser.add_argument('--retry-ranges', type=str, 
                      help='Comma-separated list of start-end ranges to retry (e.g., "0-100,200-300")')
    parser.add_argument('--json-path', type=str, default="enron_paragraphs.json",
                      help='Path to JSON file containing paragraphs')
    parser.add_argument('--output-dir', type=str, default="paragraph_embeddings",
                      help='Directory to save embedding chunks')
    parser.add_argument('--chunk-size', type=int, default=100,
                      help='Number of paragraphs per chunk')
    args = parser.parse_args()
    
    # Process specific ranges if retry requested
    specific_ranges = None
    if args.retry_ranges:
        specific_ranges = []
        for range_str in args.retry_ranges.split(','):
            start, end = map(int, range_str.split('-'))
            specific_ranges.append((start, end))
        print(f"Retrying specific ranges: {specific_ranges}")
    
    # Process chunks and get failures
    failed_ranges = process_paragraphs_in_chunks(
        args.json_path,
        args.output_dir,
        args.chunk_size,
        specific_ranges
    )
    
    # Report results
    if failed_ranges:
        print("\nFailed ranges:", failed_ranges)
        print("\nTo retry failed ranges, run:")
        ranges_str = ','.join(f"{start}-{end}" for start, end in failed_ranges)
        print(f"python compute_embeddings.py --retry-ranges {ranges_str}")
        
        # Save failed ranges for later retry
        with open(Path(args.output_dir) / "failed_ranges.txt", "w") as f:
            json.dump({
                'failed_ranges': failed_ranges,
                'chunk_size': args.chunk_size
            }, f, indent=2)
    else:
        print("\nAll ranges completed successfully!")

if __name__ == "__main__":
    main()