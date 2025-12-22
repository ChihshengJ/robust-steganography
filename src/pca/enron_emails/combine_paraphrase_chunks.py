import json
from pathlib import Path
import re
import argparse

def combine_paraphrase_chunks(chunks_dir, output_file):
    """
    Combines all paraphrase chunks in order.
    
    Args:
    - chunks_dir (str): Directory containing chunk files
    - output_file (str): Path to save combined JSON
    """
    chunks_dir = Path(chunks_dir)
    
    # Get all chunk files and their ranges
    chunk_info = []
    pattern = re.compile(r'paraphrases_(\d+)-(\d+)\.json')
    
    for chunk_file in chunks_dir.glob("paraphrases_*.json"):
        match = pattern.match(chunk_file.name)
        if not match:
            continue
            
        start_idx = int(match.group(1))
        end_idx = int(match.group(2))
        
        with open(chunk_file, 'r') as f:
            chunk_data = json.load(f)
            paragraphs = chunk_data['paragraphs']
            
        # Verify the chunk size matches the filename
        if len(paragraphs) != end_idx - start_idx:
            raise ValueError(f"Chunk {chunk_file} contains {len(paragraphs)} paragraphs but filename indicates {end_idx - start_idx}")
        
        chunk_info.append({
            'file': chunk_file,
            'paragraphs': paragraphs,
            'start_idx': start_idx,
            'end_idx': end_idx
        })
    
    if not chunk_info:
        raise ValueError(f"No paraphrase chunks found in {chunks_dir}")
    
    # Sort chunks by start index
    chunk_info.sort(key=lambda x: x['start_idx'])
    
    # Verify no gaps or overlaps
    for i in range(len(chunk_info) - 1):
        current_end = chunk_info[i]['end_idx']
        next_start = chunk_info[i + 1]['start_idx']
        
        if current_end != next_start:
            raise ValueError(f"Gap or overlap between chunks: {chunk_info[i]['file']} and {chunk_info[i + 1]['file']}")
    
    # Combine paragraphs
    all_paragraphs = []
    for chunk in chunk_info:
        all_paragraphs.extend(chunk['paragraphs'])
    
    # Save combined file
    output_data = {"paragraphs": all_paragraphs}
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Combined {len(chunk_info)} chunks into {output_file}")
    print(f"Total paragraphs: {len(all_paragraphs)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Combine paraphrase chunks')
    parser.add_argument('--chunks-dir', type=str, default="paragraph_paraphrases",
                      help='Directory containing paraphrase chunks')
    parser.add_argument('--output', type=str, default="enron_paragraph_paraphrases.json",
                      help='Path to save combined JSON')
    args = parser.parse_args()
    
    combine_paraphrase_chunks(args.chunks_dir, args.output) 