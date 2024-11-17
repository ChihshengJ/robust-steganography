import unittest
import random
from error_correction import RepetitionCode, ConvolutionalCode

class TestErrorCorrection(unittest.TestCase):
    def setUp(self):
        # Initialize with coprime values for repetitions and block_size
        self.block_sizes = [1, 2, 4, 8, 16, 32]
        self.rep_codes = {
            size: RepetitionCode(repetitions=5, block_size=size) 
            for size in self.block_sizes
        }
        self.conv_codes = {
            size: ConvolutionalCode(block_size=size) 
            for size in self.block_sizes
        }
        self.rep_error_rate = 0.05  # 5% error rate for repetition code
        self.conv_error_rate = 0.07  # 7% error rate for convolutional code
        
    def test_message_reconstruction(self):
        """Test that decode(encode(message)) == message for various message lengths"""
        test_messages = [
            [1, 0, 1],  # Short message
            [1] * 10,   # Medium message
            [random.randint(0, 1) for _ in range(50)]  # Long random message
        ]
        
        for message in test_messages:
            for block_size in self.block_sizes:
                # Test RepetitionCode
                encoded = self.rep_codes[block_size].encode(message.copy())
                decoded = self.rep_codes[block_size].decode(encoded)
                self.assertEqual(message, decoded, 
                    f"RepetitionCode failed with message length {len(message)}, block_size {block_size}")
                
                # Test ConvolutionalCode
                encoded = self.conv_codes[block_size].encode(message.copy())
                decoded = self.conv_codes[block_size].decode(encoded)
                self.assertEqual(message, decoded,
                    f"ConvolutionalCode failed with message length {len(message)}, block_size {block_size}")
    
    def test_block_size_alignment(self):
        """Test that encoded message length is always multiple of block_size"""
        test_messages = [
            [1, 0, 1],  # Short message
            [1] * 10,   # Medium message
            [random.randint(0, 1) for _ in range(50)]  # Long random message
        ]
        
        for message in test_messages:
            for block_size in self.block_sizes:
                with self.subTest(message=message, block_size=block_size):
                    # Test RepetitionCode
                    encoded = self.rep_codes[block_size].encode(message.copy())
                    self.assertEqual(len(encoded) % block_size, 0,
                        f"RepetitionCode encoded length {len(encoded)} not multiple of {block_size}")
                    
                    # Test ConvolutionalCode
                    encoded = self.conv_codes[block_size].encode(message.copy())
                    self.assertEqual(len(encoded) % block_size, 0,
                        f"ConvolutionalCode encoded length {len(encoded)} not multiple of {block_size}")
    
    def test_error_correction(self):
        """Test error correction capabilities with introduced errors"""
        # Longer message for better error correction testing
        message = [random.randint(0, 1) for _ in range(100)]
        block_size = 8
        
        # Test RepetitionCode
        encoded = self.rep_codes[block_size].encode(message.copy())
        corrupted = self._introduce_errors(encoded, error_rate=self.rep_error_rate)
        decoded = self.rep_codes[block_size].decode(corrupted)
        self.assertEqual(message, decoded,
            "RepetitionCode failed to correct errors")
        
        # Test ConvolutionalCode
        encoded = self.conv_codes[block_size].encode(message.copy())
        corrupted = self._introduce_errors(encoded, error_rate=self.conv_error_rate)
        decoded = self.conv_codes[block_size].decode(corrupted)
        self.assertEqual(message, decoded,
            "ConvolutionalCode failed to correct errors")

    def _introduce_errors(self, bits, error_rate):
        """
        Randomly flip bits based on error_rate
        
        Args:
            bits (list): List of bits to corrupt
            error_rate (float): Probability of flipping each bit (0.0 to 1.0)
        
        Returns:
            list: Corrupted bits with random errors
        """
        corrupted = bits.copy()
        num_errors = int(len(corrupted) * error_rate)  # Calculate expected number of errors
        indices = random.sample(range(len(corrupted)), num_errors)  # Choose random positions
        
        for i in indices:
            corrupted[i] = 1 - corrupted[i]  # Flip the bit
        
        return corrupted
    
    def test_edge_cases(self):
        """Test edge cases like empty messages and single-bit messages"""
        edge_cases = [
            [],     # Empty message
            [1],    # Single bit
            [0],    # Single zero
            [0]*5,  # All zeros
            [1]*5,  # All ones
        ]
        
        for message in edge_cases:
            for block_size in self.block_sizes:
                # Skip empty message if block_size is too small to hold padding information
                if not message and block_size < 4:
                    continue
                    
                # Test RepetitionCode
                try:
                    encoded = self.rep_codes[block_size].encode(message.copy())
                    decoded = self.rep_codes[block_size].decode(encoded)
                    self.assertEqual(message, decoded,
                        f"RepetitionCode failed with edge case {message}, block_size {block_size}")
                    
                    # Verify block size alignment
                    self.assertEqual(len(encoded) % block_size, 0,
                        f"RepetitionCode encoded length not multiple of {block_size}")
                except Exception as e:
                    self.fail(f"RepetitionCode failed with edge case {message}, block_size {block_size}: {str(e)}")
                
                # Test ConvolutionalCode
                try:
                    encoded = self.conv_codes[block_size].encode(message.copy())
                    decoded = self.conv_codes[block_size].decode(encoded)
                    self.assertEqual(message, decoded,
                        f"ConvolutionalCode failed with edge case {message}, block_size {block_size}")
                    
                    # Verify block size alignment
                    self.assertEqual(len(encoded) % block_size, 0,
                        f"ConvolutionalCode encoded length not multiple of {block_size}")
                except Exception as e:
                    self.fail(f"ConvolutionalCode failed with edge case {message}, block_size {block_size}: {str(e)}")

    def test_coprime_requirement(self):
        """Test that non-coprime repetitions and block_size are rejected"""
        invalid_pairs = [
            (4, 8),   # Both powers of 2
            (9, 6),   # Share factor 3
            (15, 10), # Share factor 5
        ]
        
        for rep, block in invalid_pairs:
            with self.assertRaises(ValueError) as context:
                RepetitionCode(repetitions=rep, block_size=block)
            self.assertIn("coprime", str(context.exception))

    def test_valid_parameters(self):
        """Test that coprime repetitions and block_size are accepted"""
        valid_pairs = [
            (3, 4),
            (5, 8),
            (7, 15),
        ]
        
        for rep, block in valid_pairs:
            try:
                RepetitionCode(repetitions=rep, block_size=block)
            except ValueError as e:
                self.fail(f"Valid pair ({rep}, {block}) raised ValueError: {str(e)}")

if __name__ == '__main__':
    unittest.main()