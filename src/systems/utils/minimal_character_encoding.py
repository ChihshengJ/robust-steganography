# Define the character set (32 characters total)
CHAR_TO_BITS = {
    'a': '00000', 'b': '00001', 'c': '00010', 'd': '00011', 'e': '00100',
    'f': '00101', 'g': '00110', 'h': '00111', 'i': '01000', 'j': '01001',
    'k': '01010', 'l': '01011', 'm': '01100', 'n': '01101', 'o': '01110',
    'p': '01111', 'q': '10000', 'r': '10001', 's': '10010', 't': '10011',
    'u': '10100', 'v': '10101', 'w': '10110', 'x': '10111', 'y': '11000',
    'z': '11001', ' ': '11010', '.': '11011', ',': '11100', '!': '11101',
    '?': '11110', "'": '11111'
}

# Create reverse mapping for decoding
BITS_TO_CHAR = {bits: char for char, bits in CHAR_TO_BITS.items()}

def encode_char(char):
    """Convert a single character to its 5-bit representation."""
    if char.lower() not in CHAR_TO_BITS:
        raise ValueError(f"Character '{char}' not in minimal character set")
    return [int(bit) for bit in CHAR_TO_BITS[char.lower()]]

def decode_bits(bits):
    """Convert 5 bits back to a character."""
    if len(bits) != 5:
        raise ValueError("Must provide exactly 5 bits")
    bit_string = ''.join(str(bit) for bit in bits)
    if bit_string not in BITS_TO_CHAR:
        raise ValueError(f"Invalid bit sequence: {bit_string}")
    return BITS_TO_CHAR[bit_string]

def string_to_bits(s):
    """Convert a string to a list of bits using the minimal encoding."""
    return [bit for char in s for bit in encode_char(char)]

def bits_to_string(bits):
    """Convert a list of bits back to a string."""
    if len(bits) % 5 != 0:
        raise ValueError("Number of bits must be divisible by 5")
    return ''.join(decode_bits(bits[i:i+5]) for i in range(0, len(bits), 5))
