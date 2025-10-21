from abc import ABC, abstractmethod
from typing import Any

class Encoder(ABC):
    """Base class for all encoding systems"""
    @abstractmethod
    def encode(self, data: Any) -> list[int]:
        """Convert input data to bits"""
        pass
    
    @abstractmethod
    def decode(self, bits: list[int]) -> Any:
        """Convert bits back to original data type"""
        pass

class CharacterEncoder(Encoder):
    """Base class for character-based encodings"""
    def encode(self, data: Any) -> list[int]:
        """
        Encode a string into bits.
        accept Any but will raise a TypeError if the data is not a string.
        """
        if not isinstance(data, str):
            raise TypeError(
                f"CharacterEncoder can only encode strings, but received type '{type(data).__name__}'"
            )

        text: str = data
        bits = []
        for char in text:
            # Using 8-bit ASCII for this example
            binary_representation = bin(ord(char))[2:].zfill(8)
            bits.extend([int(b) for b in binary_representation])
        return bits

    def decode(self, bits: list[int]) -> str:
        if len(bits) % 8 != 0:
            raise ValueError("Bit list length must be a multiple of 8.")
        
        chars = []
        for i in range(0, len(bits), 8):
            byte_string = "".join(map(str, bits[i:i+8]))
            char_code = int(byte_string, 2)
            chars.append(chr(char_code))
            
        return "".join(chars)
    

class StandardEncoder(CharacterEncoder):
    BITS_PER_CHAR = 8  # UTF-8/ASCII uses 8 bits per character
    
    def encode(self, data: str) -> list[int]:
        return [int(bit) for char in data for bit in format(ord(char), '08b')]
    
    def decode(self, bits: list[int]) -> str:
        """Convert bits back to string, with error checking."""
        if len(bits) % self.BITS_PER_CHAR != 0:
            raise ValueError(
                f"Received {len(bits)} bits which is not divisible by {self.BITS_PER_CHAR}. "
                "This likely means the error correction padding field was corrupted. "
                "Too many errors occurred to successfully decode the message."
            )
        chars = []
        for b in range(0, len(bits), self.BITS_PER_CHAR):
            byte = bits[b:b+self.BITS_PER_CHAR]
            char = chr(int(''.join(map(str, byte)), 2))
            chars.append(char)
        return ''.join(chars)

class MinimalEncoder(CharacterEncoder):
    BITS_PER_CHAR = 5  # From CHAR_TO_BITS mapping
    
    def __init__(self):
        from ..utils.minimal_character_encoding import string_to_bits, bits_to_string
        self.encode_text = string_to_bits
        self.decode_bits = bits_to_string
    
    def encode(self, data: str) -> list[int]:
        return self.encode_text(data)
    
    def decode(self, bits: list[int]) -> str:
        """Decode a bit sequence into a string, with error checking."""
        if len(bits) % self.BITS_PER_CHAR != 0:
            raise ValueError(
                f"Received {len(bits)} bits which is not divisible by {self.BITS_PER_CHAR}. "
                "This likely means the error correction padding field was corrupted. "
                "Too many errors occurred to successfully decode the message."
            )
        return self.decode_bits(bits)

class CiphertextEncoder(Encoder):
    """
    Handles encoding/decoding of ciphertext objects by serializing them to a standard format.
    Supports any encryption object that can be serialized to JSON or bytes.
    
    When using 'bytes' format, this encoder can also handle raw binary data directly:
        - Raw bytes (e.g., b"Hello World")
        - Bytearray objects
        - Any object that can be converted to bytes
        - UTF-8 encoded strings (after calling .encode('utf-8'))
    
    Examples:
        # For encrypted data
        encoder = CiphertextEncoder('bytes')
        bits = encoder.encode(cipher.encrypt(b"secret"))
        
        # For raw binary data
        encoder = CiphertextEncoder('bytes')
        bits = encoder.encode(b"raw bytes")
    """
    def __init__(self, serialization_format: str = 'json'):
        """
        Args:
            serialization_format: Either 'json' or 'bytes' depending on encryption library
                                Use 'bytes' for raw binary data
        """
        import json
        import base64
        self.format = serialization_format
        self.json = json
        self.base64 = base64
        
    def encode(self, data: Any) -> list[int]:
        if self.format == 'json':
            # Directly use the dictionary if it's already JSON-serializable
            if isinstance(data, dict):
                data = data
            elif hasattr(data, 'to_dict'):
                data = data.to_dict()
            else:
                data = data.__dict__
            
            # Convert to JSON string then to bits
            json_str = self.json.dumps(data)
            return StandardEncoder().encode(json_str)
            
        elif self.format == 'bytes':
            # Handle byte-based ciphertext
            if isinstance(data, bytes):
                byte_data = data
            else:
                byte_data = bytes(data)
                
            # Convert bytes to base64 string then to bits
            b64_str = self.base64.b64encode(byte_data).decode('utf-8')
            return StandardEncoder().encode(b64_str)
            
        else:
            raise ValueError(f"Unsupported serialization format: {self.format}")
    
    def decode(self, bits: list[int]) -> Any:
        try:
            text = StandardEncoder().decode(bits)
        except ValueError as e:
            raise ValueError(f"Failed to decode ciphertext: {str(e)}") from e
        
        if self.format == 'json':
            try:
                return self.json.loads(text)
            except self.json.JSONDecodeError:
                raise ValueError(
                    "Failed to parse JSON from decoded bits. "
                    "This likely means too many errors occurred during transmission."
                )
            
        elif self.format == 'bytes':
            try:
                return self.base64.b64decode(text.encode('utf-8'))
            except Exception:
                raise ValueError(
                    "Failed to decode base64 from decoded bits. "
                    "This likely means too many errors occurred during transmission."
                )
        
        else:
            raise ValueError(f"Unsupported serialization format: {self.format}")
