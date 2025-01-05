import json
from pathlib import Path
from tqdm import tqdm
from robust_steganography.utils.paraphrase import paraphrase_message
from openai import OpenAI
import argparse
import nltk

def process_paragraphs_in_chunks(json_path, output_dir, chunk_size=100, specific_ranges=None):
    """
    Process paragraphs from JSON file in chunks and save their paraphrases.
    
    Args:
    - json_path (str): Path to original JSON file
    - output_dir (str): Directory to save paraphrase chunks
    - chunk_size (int): Number of paragraphs per chunk
    - specific_ranges (list): List of (start, end) tuples for specific ranges to process
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load paragraphs from JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        paragraphs = data['paragraphs']
    
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
        output_path = output_dir / f'paraphrases_{start_idx:06d}-{end_idx:06d}.json'
        
        # Skip if chunk already exists
        if output_path.exists():
            print(f"Range {start_idx}-{end_idx} already exists, skipping...")
            continue
        
        # Get chunk of paragraphs
        chunk_paragraphs = paragraphs[start_idx:end_idx]
        paraphrased_paragraphs = []
        
        try:
            # Process each paragraph in chunk
            for para in chunk_paragraphs:
                try:
                    paraphrased_text = paraphrase_message(client, para['text'])
                    paraphrased_para = {
                        "text": paraphrased_text,
                        "sentence_count": len(nltk.sent_tokenize(paraphrased_text)),
                        "sentences": nltk.sent_tokenize(paraphrased_text),
                        "word_count": len(paraphrased_text.split())
                    }
                    paraphrased_paragraphs.append(paraphrased_para)
                except Exception as e:
                    print(f"Error paraphrasing paragraph: {str(e)}")
                    paraphrased_paragraphs.append(None)  # Keep alignment with original
            
            # Verify we processed the expected number of paragraphs
            if len(paraphrased_paragraphs) != len(chunk_paragraphs):
                raise ValueError(f"Got {len(paraphrased_paragraphs)} paraphrases for {len(chunk_paragraphs)} paragraphs")
            
            # Save chunk
            chunk_data = {"paragraphs": paraphrased_paragraphs}
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(chunk_data, f, indent=2)
            print(f"Saved range {start_idx}-{end_idx}")
            
        except Exception as e:
            print(f"Error processing range {start_idx}-{end_idx}: {str(e)}")
            failed_ranges.append((start_idx, end_idx))
            continue
    
    return failed_ranges

def main():
    parser = argparse.ArgumentParser(description='Create paraphrases of Enron paragraphs')
    parser.add_argument('--retry-ranges', type=str, 
                      help='Comma-separated list of start-end ranges to retry (e.g., "0-100,200-300")')
    parser.add_argument('--json-path', type=str, default="enron_paragraphs.json",
                      help='Path to JSON file containing paragraphs')
    parser.add_argument('--output-dir', type=str, default="paragraph_paraphrases",
                      help='Directory to save paraphrase chunks')
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
        print(f"python create_paraphrases.py --retry-ranges {ranges_str}")
        
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