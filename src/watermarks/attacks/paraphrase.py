from openai import OpenAI
import re

class ParaphraseAttack:
    """Attack that uses GPT to paraphrase text while preserving meaning."""
    
    def __init__(self, client: OpenAI, model: str = "gpt-4o-mini", temperature: float = 0.0, local: bool = True):
        """
        Initialize the paraphrase attack.
        
        Args:
            client: OpenAI client instance
            model: GPT model to use (default: "gpt-4o-mini")
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            local: If True, paraphrases each sentence independently.
                  If False, paraphrases entire text at once (default: True)
        """
        self.client = client
        self.model = model
        self.temperature = temperature
        self.local = local
        
        self.system_prompt = """
        You are an assistant tasked with paraphrasing text. For each input, your goal 
        is to rephrase it using different wording and sentence structures while ensuring 
        that the original meaning, intent, and nuances are completely preserved. Do not 
        omit or add new information. The paraphrased output should be clear, concise, 
        and faithful to the original message.
        
        Important: Preserve all formatting including newlines, spaces, and punctuation 
        placement. Return only the paraphrased text with no additional commentary.
        """

    def __call__(self, text: str) -> str:
        """Apply the paraphrase attack."""
        if self.local:
            return self._local_paraphrase(text)
        else:
            return self._global_paraphrase(text)
            
    def _global_paraphrase(self, text: str) -> str:
        """Paraphrase entire text at once."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": text}
                ],
                temperature=self.temperature,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Global paraphrase attack failed: {e}")
            return text
            
    def _local_paraphrase(self, text: str) -> str:
        """Paraphrase each sentence independently while preserving structure."""
        # Split text into sentences while preserving separators
        parts = re.split(r'([.!?]+(?:\s+|$))', text)
        new_parts = []
        
        # parts[::2] are sentences, parts[1::2] are separators
        for i in range(0, len(parts), 2):
            sentence = parts[i]
            
            # Skip empty sentences
            if not sentence.strip():
                new_parts.append(sentence)
            else:
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": sentence}
                        ],
                        temperature=self.temperature,
                        top_p=1.0,
                        frequency_penalty=0.0,
                        presence_penalty=0.0
                    )
                    new_parts.append(response.choices[0].message.content.strip())
                except Exception as e:
                    print(f"Local paraphrase attack failed for sentence: {e}")
                    new_parts.append(sentence)
            
            # Add the separator if it exists
            if i + 1 < len(parts):
                new_parts.append(parts[i + 1])
        
        return ''.join(new_parts) 