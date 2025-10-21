import json
from pathlib import Path
from typing import List, Dict

def log_prf_output(
    mode: str,  # 'embed' or 'extract'
    step: int,  # Current token position
    prf_output: List[int],
    context_tokens: List[int],
    salt: int,
    key: bytes
) -> None:
    """
    Log PRF outputs and related info for debugging.
    
    Parameters
    ----------
    mode : str
        Whether this is from embedder or extractor
    step : int
        Current token position being processed
    prf_output : List[int]
        Binary labels from PRF
    context_tokens : List[int]
        Token IDs used as context for PRF
    salt : int
        Salt value used
    key : bytes
        PRF key used (will be converted to hex)
    """
    log_dir = Path("debug_logs")
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"prf_{mode}_log.jsonl"
    
    entry = {
        "step": step,
        "context_length": len(context_tokens),
        "context_tokens": context_tokens,
        "salt": salt,
        "key": key.hex(),  # Convert bytes to hex string
        "prf_output_sum": sum(prf_output),  # For quick comparison
        "prf_output": prf_output,
    }
    
    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n") 