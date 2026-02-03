import re
import datetime
from typing_extensions import override
from typing import Any

import numpy as np
from nltk.tokenize import sent_tokenize


from ..config.constants import BacktrackConfig

# Import the NEW prompts
from ..config.system_prompts import (
    FACT_CONTINUATION_ANCHORED,
    FACT_GENERATION_ANCHORED,
    FACT_SUMMARY_STRICT,
    FACT_EXTRACT_FROM_SUMMARY
)
from .steg_system import StegSystem
from ..utils.get_embedding import get_embeddings_in_batch
from ..utils.new_text import generate_response
from ..utils.sample_utils import BacktrackingEncoder, RejectionSampler, SampleResult
from .encoder import Encoder
from .error_correction import ErrorCorrection
from .hash_functions import HashFunction


class SummarySystemV2(StegSystem):
    """
    Improved summary-based steganography system.
    
    Key improvements:
    - Semantic anchoring in fact generation for stable embeddings
    - Sentence-preserving summary generation
    - Sentence-level recovery using NLTK (no LLM decomposition)
    - FIXED: Proper history building that includes all previously generated facts
    """
    
    def __init__(
        self,
        client,
        key: int,
        hash_function: HashFunction,
        error_correction: ErrorCorrection,
        encoder: Encoder | None = None,
        backtrack_config: BacktrackConfig | None = None,
    ):
        if key < 0:
            raise ValueError("key must be non-negative")
        super().__init__(client, hash_function, error_correction, encoder)
        self.message_length: int | None = None
        self.key = key
        self.hash_output_length = getattr(hash_function, "output_length")
        self.backtrack_config = backtrack_config or BacktrackConfig()
        
        from ..utils.sample_utils import RejectionSampler
        self._fact_sampler = FactAwareSampler()
        self._fact_encoder = FactAwareBacktrackingEncoder(
            sampler=self._fact_sampler,
            config=self.backtrack_config,
        )
        
        self._base_facts: list[str] = []
        self._optional_facts: list[str] = []

    @override
    def encode(
        self,
        chunks: list[list[int]],
        article: str,
        base_facts: list[str],
        system_prompt: str,
        max_length: int = 200,
        temperature: float = 1.0,
        **kwargs,
    ) -> tuple[list[str], list]:
        """Encode chunks into facts using fact-aware rejection sampling."""
        facts, embeddings = self._fact_encoder.encode(
            client=self.client,
            chunks=[np.array(lst) for lst in chunks],
            article=article,
            base_facts=base_facts,
            hash_fn=self.hash_fn,
            system_prompt=system_prompt,
            max_length=max_length,
            temperature=temperature,
            use_prohibitions=True,
        )
        return facts, embeddings

    def hide_message(self, data: Any, seed: str, **kwargs) -> str:
        chunks, self.message_length = self._encode_to_chunks(data)

        # Step 1: Generate base facts (semantic anchoring)
        base_facts_raw = generate_response(
            client=self.client,
            prompt=seed,
            system_prompt=FACT_GENERATION_ANCHORED.format(k=self.key),
            max_length=1500,
            temperature=0,
        )
        
        # Parse base facts
        self._base_facts = self._parse_numbered_facts(base_facts_raw)
        print(f"\n{'='*60}")
        print(f"Base facts ({len(self._base_facts)}):")
        print(f"{'='*60}")
        for i, f in enumerate(self._base_facts):
            print(f"  {f[:100]}...")

        # Step 2: Generate optional facts via rejection sampling
        print(f"\n{'='*60}")
        print(f"Generating {len(chunks)} optional facts...")
        print(f"{'='*60}")
        
        self._optional_facts, _ = self.encode(
            chunks=chunks,
            article=seed,
            base_facts=self._base_facts,
            system_prompt=FACT_CONTINUATION_ANCHORED,
            max_length=200,
            temperature=1.0,
        )
        
        print(f"\n{'='*60}")
        print(f"Optional facts ({len(self._optional_facts)}):")
        print(f"{'='*60}")
        for i, f in enumerate(self._optional_facts):
            print(f"  {self.key + i + 1}. {f[:100]}...")

        # Step 3: Generate sentence-preserving summary
        all_facts = self._base_facts + self._optional_facts
        facts_formatted = "\n".join(f"{i+1}. {fact}" for i, fact in enumerate(all_facts))
        
        print(f"\n{'='*60}")
        print("Generating summary...")
        print(f"{'='*60}")
        
        stego_text = generate_response(
            client=self.client,
            prompt=facts_formatted,
            system_prompt=FACT_SUMMARY_STRICT,
            max_length=2000,
            temperature=0,
        ).strip()
        
        print(f"\nGenerated summary ({len(sent_tokenize(stego_text))} sentences):")
        print(f"  {stego_text[:300]}...")

        # Verify alignment
        self._verify_alignment(stego_text, all_facts)

        return stego_text

    def recover_message(self, stego_text: str):
        """
        Recover the encoded message from a summary using LLM-based decomposition.
        
        Uses anchor-aware fact extraction to reliably split the summary back into
        individual facts, handling complex quote structures correctly.
        """
        if self.message_length is None:
            raise ValueError(
                "No message length set. Run hide_message first or set message_length."
            )

        expected_optional = self.message_length // self.hash_output_length
        expected_total = self.key + expected_optional
        
        print(f"\n{'='*60}")
        print("Recovery - LLM Decomposition")
        print(f"{'='*60}")
        print(f"  Expected facts: {expected_total} (key={self.key}, optional={expected_optional})")

        # Use LLM to decompose summary into facts
        all_facts = self._decompose_summary_llm(stego_text, expected_total)
        
        print(f"  Extracted facts: {len(all_facts)}")
        
        if len(all_facts) != expected_total:
            print(f"    WARNING: Expected {expected_total} facts, got {len(all_facts)}")
        
        # Extract optional facts (after base facts)
        optional_facts = all_facts[self.key:self.key + expected_optional]
        
        print(f"\n  Optional facts for decoding ({len(optional_facts)}):")
        for i, f in enumerate(optional_facts):
            print(f"    {self.key + i + 1}. {f[:70]}...")

        if len(optional_facts) < expected_optional:
            print(f"    Only {len(optional_facts)} optional facts, expected {expected_optional}")

        embeddings = get_embeddings_in_batch(self.client, optional_facts)
        return self._decode_from_embeddings(embeddings, self.message_length)


    def _decompose_summary_llm(self, summary: str, num_facts: int) -> list[str]:
        """
        Use LLM to decompose summary into individual facts.
        Leverages semantic anchoring to identify fact boundaries.
        """
        response = generate_response(
            client=self.client,
            prompt=summary,
            system_prompt=FACT_EXTRACT_FROM_SUMMARY.format(num_facts=num_facts),
            max_length=2000,
            temperature=0,  # Deterministic extraction
        )
        
        print("\n  Decomposition response preview:")
        print(f"    {response[:300]}...")
        
        facts = self._parse_sep_delimited_facts(response)
        
        return facts


    def _parse_sep_delimited_facts(self, response: str) -> list[str]:
        """Parse facts delimited by [sep] markers."""
        facts = []
        parts = response.split("[sep]")
        
        for part in parts:
            fact = part.strip()
            if fact:
                fact = re.sub(r"^\d+[.):\s]+", "", fact).strip()
                fact = re.sub(r"^[-•*]\s*", "", fact).strip()
                if fact:
                    facts.append(fact)
        
        return facts


    def _parse_numbered_facts(self, raw_text: str) -> list[str]:
        """Parse numbered facts from LLM output."""
        facts = []
        lines = raw_text.strip().split("\n")
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            for i in range(1, 100):
                prefix = f"{i}."
                if line.startswith(prefix):
                    fact_text = line[len(prefix):].strip()
                    if fact_text:
                        facts.append(fact_text)
                    break
        
        return facts

    def _verify_alignment(self, summary: str, facts: list[str]) -> None:
        """Verify that summary sentences align with facts."""
        sentences = sent_tokenize(summary)
        
        print("\n  Alignment check:")
        if len(sentences) != len(facts):
            print(f"    MISMATCH: {len(sentences)} sentences vs {len(facts)} facts")
        else:
            print(f"    OK: {len(sentences)} sentences = {len(facts)} facts")



def test_tokenization(text: str) -> list[str]:
    """Test how a text will be tokenized for recovery."""
    sentences = sent_tokenize(text)
    print(f"Tokenized into {len(sentences)} sentences:")
    for i, s in enumerate(sentences):
        print(f"  {i + 1}. {s}")
    return sentences


class FactAwareSampler:
    def sample(
        self,
        client: Any,
        desired_bits: np.ndarray,
        history: dict,
        hash_fn: HashFunction,
        system_prompt: str,
        max_length: int,
        temperature: float = 0.7,
        max_attempts: int = 50,
        collect_alternatives: bool = True,
        max_alternatives: int = 3,
        **kwargs,
    ) -> SampleResult:
        matches = []
        attempts = 0
        
        covered_facts: set[str] = set()
        
        while attempts < max_attempts:
            prompt = self._build_prompt(
                article=history['article'],
                base_facts=history['base_facts'],
                optional_facts=history['optional_facts'],
                failed_attempts=covered_facts,
            )
            
            response = generate_response(
                client,
                prompt,
                system_prompt,
                max_length,
                temperature,
            )
            attempts += 1
            
            response = self._clean_response(
                response, 
                len(history['base_facts']) + len(history['optional_facts']) + len(covered_facts) + 1
            )
            
            all_existing = (
                set(history['base_facts']) | 
                set(history['optional_facts']) | 
                covered_facts
            )
            
            if response in all_existing:
                # print(f"  [Attempt {attempts}] Exact duplicate, already in covered list")
                continue
            
            if self._is_near_duplicate(response, all_existing):
                # print(f"  [Attempt {attempts}] Near-duplicate detected, adding to covered")
                covered_facts.add(response)
                continue
            
            embeddings = get_embeddings_in_batch(client, [response])
            embedding = np.array(embeddings[0]).reshape(1, -1)
            sampled_bits = hash_fn(embedding)
            
            # print(f"  [Attempt {attempts}] Response: {response[:60]}...")
            # print(f"  [Attempt {attempts}] desired: {desired_bits}, sampled: {sampled_bits}")
            
            if np.array_equal(sampled_bits, desired_bits):
                matches.append((response, embedding))
                covered_facts.add(response)  # Prevent finding same match
                
                if not collect_alternatives or len(matches) >= max_alternatives:
                    return SampleResult(
                        success=True,
                        matches=matches,
                        attempts_used=attempts,
                    )
            else:
                covered_facts.add(response)
                # print(f"  [Attempt {attempts}] Hash mismatch, added to covered facts ({len(covered_facts)} total)")
        
        return SampleResult(
            success=len(matches) > 0,
            matches=matches,
            attempts_used=attempts,
        )
    
    def _build_prompt(
        self,
        article: str,
        base_facts: list[str],
        optional_facts: list[str],
        failed_attempts: set[str],
    ) -> str:
        """
        Build prompt where failed attempts appear as "already extracted" facts.
        
        This is the key reframing: instead of "don't generate these",
        we say "these are already covered, find something new".
        """
        all_covered = list(base_facts) + list(optional_facts) + list(failed_attempts)
        
        facts_numbered = "\n".join(f"{i+1}. {f}" for i, f in enumerate(all_covered))
        
        topic_hints = self._extract_topics(all_covered)
        
        prompt = f"""{article}
            FACTS ALREADY EXTRACTED ({len(all_covered)} total) - you must find something DIFFERENT than these:
            {facts_numbered}

            TOPICS ALREADY COVERED: {topic_hints}

            Extract 1 NEW fact that:
            - Covers a DIFFERENT detail than all {len(all_covered)} facts above
            - Is a single complete sentence
            - Contains specific names, numbers, or dates from the article

            Look for: secondary details, specific quotes, background context, reactions, locations, times.

            What specific detail from the article is NOT yet captured above?
        """
            
        return prompt
    
    def _extract_topics(self, facts: list[str]) -> str:
        """Extract key topics to show what's been covered."""
        keywords = set()
        
        for fact in facts:
            words = fact.split()
            for word in words:
                if word and word[0].isupper() and len(word) > 3:
                    clean = word.strip('.,;:"\'-()[]')
                    if clean and clean[0].isupper():
                        keywords.add(clean)
        
        if keywords:
            top_keywords = sorted(keywords)[:8]
            return ", ".join(top_keywords)
        
        return "main event details"
    
    def _clean_response(self, response: str, expected_num: int) -> str:
        """Clean response to extract just the fact text."""
        import re
        
        response = response.strip()
        
        patterns = [
            rf"^{expected_num}[.):\s]+",
            r"^\d+[.):\s]+",
            r"^[Ff]act[:\s]+",
            r"^[Nn]ew [Ff]act[:\s]+",
            r"^[-•*]\s*",
        ]
        
        for pattern in patterns:
            response = re.sub(pattern, "", response).strip()
        
        response = response.strip('"\'')
        
        return response
    
    def _is_near_duplicate(self, new_fact: str, existing: set[str], threshold: float = 0.8) -> bool:
        """
        Check if new fact is too similar to existing facts.
        Uses simple word overlap - could be enhanced with embeddings.
        """
        new_words = set(new_fact.lower().split())
        
        for existing_fact in existing:
            existing_words = set(existing_fact.lower().split())
            
            if not new_words or not existing_words:
                continue
            
            intersection = len(new_words & existing_words)
            union = len(new_words | existing_words)
            
            if union > 0 and intersection / union > threshold:
                return True
        
        return False


class FactAwareBacktrackingEncoder:
    """
    Backtracking encoder that maintains structured fact history
    instead of flat string concatenation.
    """
    
    def __init__(self, sampler, config):
        self.sampler = sampler
        self.config = config
    
    def encode(
        self,
        client: Any,
        chunks: list[np.ndarray],
        article: str,
        base_facts: list[str],
        hash_fn,
        system_prompt: str,
        max_length: int,
        temperature: float = 0.5,
        use_prohibitions: bool = True,
    ) -> tuple[list[str], list[np.ndarray]]:
        """
        Encode chunks into facts, maintaining proper fact history.
        """
        from ..utils.sample_utils import StepChoice, EncodingError, SampleResult
        
        steps: list[StepChoice] = []
        optional_facts: list[str] = []
        i = 0
        backtracks_used = 0
        
        while i < len(chunks):
            history = {
                'article': article,
                'base_facts': base_facts,
                'optional_facts': optional_facts.copy(),
            }
            
            collect_alternatives = (
                False if i == (len(chunks) - 1) else self.config.collect_alternatives
            )
            max_attempts = self.config.max_attempts_per_step
            if i == 0:
                max_attempts = int(max_attempts * 1.5)
            
            print(f"\n[Encoding fact {i + 1}/{len(chunks)}]")
            print(f"  Base facts: {len(base_facts)}, Optionals so far:\n {'\n'.join(optional_facts)}")
            
            result: SampleResult = self.sampler.sample(
                client=client,
                desired_bits=chunks[i],
                history=history,
                hash_fn=hash_fn,
                system_prompt=system_prompt,
                max_length=max_length,
                temperature=temperature,
                max_attempts=max_attempts,
                collect_alternatives=collect_alternatives,
                max_alternatives=self.config.max_alternatives,
                use_prohibitions=use_prohibitions,
            )
            
            if result.success:
                self._log_success(i, len(chunks), result)
                
                if i < len(steps):
                    steps[i] = StepChoice(result.matches)
                    optional_facts[i] = result.matches[0][0]
                else:
                    steps.append(StepChoice(result.matches))
                    optional_facts.append(result.matches[0][0])
                i += 1
            else:
                backtracks_used += 1
                if backtracks_used > self.config.max_backtracks:
                    raise EncodingError(
                        f"Exceeded maximum backtracks ({self.config.max_backtracks})"
                    )
                
                target = self._find_backtrack_target(steps)
                if target < 0:
                    raise EncodingError(
                        f"No backtrack target available at position {i}"
                    )
                
                self._log_backtrack(i, target, backtracks_used)
                steps[target].use_next_alternative()
                
                # Update optional_facts to reflect the alternative
                optional_facts[target] = steps[target].message
                
                # Truncate steps and optional_facts
                steps = steps[:target + 1]
                optional_facts = optional_facts[:target + 1]
                i = target + 1
        
        cover_texts = [choice.message for choice in steps]
        embeddings = [choice.embedding for choice in steps]
        return cover_texts, embeddings
    
    def _find_backtrack_target(self, steps) -> int:
        for i in range(len(steps) - 1, -1, -1):
            if steps[i].has_alternatives():
                return i
        return -1
    
    def _log_success(self, position: int, total: int, result) -> None:
        alt_info = ""
        if len(result.matches) > 1:
            alt_info = f" (+{len(result.matches) - 1} alternatives)"
        print(f"[{position + 1}/{total}] Encoded in {result.attempts_used} attempts{alt_info}")
    
    def _log_backtrack(self, from_pos: int, to_pos: int, total_backtracks: int) -> None:
        print(f"[Backtrack #{total_backtracks}] {from_pos + 1} -> {to_pos + 1}")
