import math
from abc import ABC, abstractmethod

import numpy as np


class ErrorCorrection(ABC):
    @abstractmethod
    def encode(self, bits):
        pass

    @abstractmethod
    def decode(self, bits):
        pass


class RepetitionCode(ErrorCorrection):
    def __init__(self, repetitions=3, block_size=1):
        if repetitions < 1:
            raise ValueError("Repetitions must be at least 1")
        if block_size < 1:
            raise ValueError("Block size must be at least 1")
        if not self._check_parameters(repetitions, block_size):
            raise ValueError("Repetitions and block_size must be coprime")
        self.repetitions = repetitions
        self.block_size = block_size

    def _check_parameters(self, repetitions, block_size):
        """Check if repetitions and block_size are coprime."""
        return math.gcd(repetitions, block_size) == 1

    def encode(self, bits):
        # Pad the input bits
        padded_bits = self._pad_input(bits)

        # Encode with repetition code
        encoded_bits = []
        for bit in padded_bits:
            encoded_bits.extend([bit] * self.repetitions)

        return encoded_bits

    def decode(self, bits):
        # Flatten input array
        bits = np.array(bits).flatten()

        decoded_bits = []
        for i in range(0, len(bits), self.repetitions):
            chunk = bits[i : i + self.repetitions]
            decoded_bits.append(int(sum(chunk) > self.repetitions // 2))

        # Unpad the decoded bits
        unpadded_bits = self._unpad_output(decoded_bits)
        return unpadded_bits

    def _pad_input(self, bits):
        """Add padding bits and padding length field to ensure encoded length is multiple of block_size"""
        if self.block_size == 1:
            return bits

        max_padding_value = self.block_size - 1
        padding_field_size = math.ceil(math.log2(max_padding_value + 1))

        # Calculate num_padding using explicit calculation
        num_padding = self._calculate_padding(
            len(bits), padding_field_size, max_padding_value
        )

        # Convert num_padding to binary and add padding
        padding_field = [int(b) for b in format(num_padding, f"0{padding_field_size}b")]
        return padding_field + bits + [0] * num_padding

    def _calculate_padding(self, bits_length, padding_field_size, max_padding_value):
        """Calculate num_padding explicitly using modular arithmetic."""
        N = bits_length + padding_field_size
        A = (N * self.repetitions) % self.block_size
        r = self.repetitions % self.block_size

        # Compute modular inverse of r modulo block_size
        try:
            r_inv = pow(r, -1, self.block_size)
        except ValueError:
            raise ValueError(
                "No modular inverse exists; check that repetitions and block_size are coprime."
            )

        # Compute num_padding
        num_padding = (-A * r_inv) % self.block_size

        # Ensure num_padding is within allowable range
        if num_padding > max_padding_value:
            raise ValueError("Cannot find appropriate padding within allowed range")

        return num_padding

    def _unpad_output(self, bits):
        if self.block_size == 1:
            return bits

        max_padding_value = self.block_size - 1
        padding_field_size = math.ceil(math.log2(max_padding_value + 1))

        # Extract padding length and remove padding
        padding_field = bits[:padding_field_size]
        num_padding = int("".join(map(str, padding_field)), 2)
        return (
            bits[padding_field_size:-num_padding]
            if num_padding > 0
            else bits[padding_field_size:]
        )


class ConvolutionalCode(ErrorCorrection):
    def __init__(self, block_size=1):
        self.K = 15  # Constraint length
        self.width = self.K
        self.block_size = block_size

        # Generator polynomials in octal
        self.g0_octal = "076461"
        self.g1_octal = "067327"
        self.g2_octal = "055611"
        self.g3_octal = "047533"

        # Convert generators to binary lists
        self.g0 = self._int_to_bin_list(self._octal_to_int(self.g0_octal))
        self.g1 = self._int_to_bin_list(self._octal_to_int(self.g1_octal))
        self.g2 = self._int_to_bin_list(self._octal_to_int(self.g2_octal))
        self.g3 = self._int_to_bin_list(self._octal_to_int(self.g3_octal))

    def _octal_to_int(self, octal_str):
        return int(octal_str, 8)

    def _int_to_bin_list(self, num):
        return [int(b) for b in format(num, f"0{self.width}b")]

    def encode(self, input_bits):
        n = 4  # Code rate denominator
        total_tail_bits = self.K - 1

        # Pad the input bits
        padded_input_bits, num_padding_bits, num_padding_bits_field_size = (
            self._pad_input(input_bits, n, self.K)
        )

        # Initialize shift register
        shift_register = [0] * self.K
        output_bits = []

        # Pad input with K-1 zeros to flush encoder
        input_bits_padded = padded_input_bits + [0] * total_tail_bits

        for bit in input_bits_padded:
            # Shift in new bit
            shift_register = [bit] + shift_register[:-1]

            # Compute output bits using generator polynomials
            o0 = sum([shift_register[i] * self.g0[i] for i in range(self.K)]) % 2
            o1 = sum([shift_register[i] * self.g1[i] for i in range(self.K)]) % 2
            o2 = sum([shift_register[i] * self.g2[i] for i in range(self.K)]) % 2
            o3 = sum([shift_register[i] * self.g3[i] for i in range(self.K)]) % 2

            output_bits.extend([o0, o1, o2, o3])

        return output_bits

    def _pad_input(self, input_bits, n, K):
        # Special case: if block_size is 1, no padding needed
        if self.block_size == 1:
            return input_bits.copy(), 0, 0

        total_tail_bits = K - 1
        L_input = len(input_bits)
        padded_input_bits = input_bits.copy()

        # Calculate bits needed to represent num_padding_bits
        max_padding_value = self.block_size - 1
        num_padding_bits_field_size = math.ceil(math.log2(max_padding_value + 1))

        # Calculate initial encoded length (including field size and tail bits)
        L_total = L_input + num_padding_bits_field_size
        encoded_length = n * (L_total + total_tail_bits)

        # Calculate padding needed at the beginning
        num_padding_bits = 0
        while encoded_length % self.block_size != 0:
            num_padding_bits += 1
            L_total = L_input + num_padding_bits_field_size + num_padding_bits
            encoded_length = n * (L_total + total_tail_bits)

        # Represent num_padding_bits as bits
        num_padding_bits_bits = [
            int(b) for b in format(num_padding_bits, f"0{num_padding_bits_field_size}b")
        ]

        # Add padding bits after the field size but before the input
        padded_input_bits = (
            num_padding_bits_bits + [0] * num_padding_bits + padded_input_bits
        )

        return padded_input_bits, num_padding_bits, num_padding_bits_field_size

    def decode(self, received_bits):
        # Flatten input array
        received_bits = np.array(received_bits).flatten()

        n = 4  # Code rate denominator
        total_tail_bits = self.K - 1

        # Viterbi algorithm implementation
        # Initialize path metrics
        num_output_bits = n
        num_time_steps = len(received_bits) // num_output_bits
        path_metrics = [{} for _ in range(num_time_steps + 1)]
        path_metrics[0][0] = {"metric": 0, "path": []}

        for time in range(num_time_steps):
            received_symbols = received_bits[
                num_output_bits * time : num_output_bits * (time + 1)
            ]
            current_metrics = {}

            for prev_state in path_metrics[time]:
                prev_metric = path_metrics[time][prev_state]["metric"]
                prev_path = path_metrics[time][prev_state]["path"]

                for input_bit in [0, 1]:
                    # Determine next state
                    prev_state_bits = [
                        int(b) for b in format(prev_state, f"0{self.K - 1}b")
                    ]
                    shift_reg = [input_bit] + prev_state_bits
                    next_state_bits = shift_reg[:-1]
                    next_state = int("".join(map(str, next_state_bits)), 2)

                    # Compute expected output
                    o0 = sum([shift_reg[i] * self.g0[i] for i in range(self.K)]) % 2
                    o1 = sum([shift_reg[i] * self.g1[i] for i in range(self.K)]) % 2
                    o2 = sum([shift_reg[i] * self.g2[i] for i in range(self.K)]) % 2
                    o3 = sum([shift_reg[i] * self.g3[i] for i in range(self.K)]) % 2
                    expected_symbols = [o0, o1, o2, o3]

                    # Compute branch metric (Hamming distance)
                    branch_metric = sum(
                        [
                            abs(received_symbols[i] - expected_symbols[i])
                            for i in range(num_output_bits)
                        ]
                    )
                    total_metric = prev_metric + branch_metric

                    # Update path metrics if better path found
                    if (
                        next_state not in current_metrics
                        or total_metric < current_metrics[next_state]["metric"]
                    ):
                        current_metrics[next_state] = {
                            "metric": total_metric,
                            "path": prev_path + [input_bit],
                        }

            path_metrics[time + 1] = current_metrics

        # Find best path
        final_state = min(path_metrics[-1], key=lambda s: path_metrics[-1][s]["metric"])
        decoded_bits = path_metrics[-1][final_state]["path"]

        # Remove tail bits
        if total_tail_bits > 0:
            decoded_bits = decoded_bits[:-(total_tail_bits)]

        # Unpad the decoded bits
        decoded_bits = self._unpad_output(decoded_bits)
        return decoded_bits

    def _unpad_output(self, decoded_bits):
        # Special case: if block_size is 1, no padding is needed
        if self.block_size == 1:
            return decoded_bits

        # Calculate bits needed to represent num_padding_bits
        max_padding_value = self.block_size - 1
        num_padding_bits_field_size = math.ceil(math.log2(max_padding_value + 1))

        # Extract num_padding_bits_bits
        num_padding_bits_bits = decoded_bits[:num_padding_bits_field_size]
        num_padding_bits = int("".join(map(str, num_padding_bits_bits)), 2)

        # Remove padding bits and padding field
        decoded_bits = decoded_bits[num_padding_bits_field_size + num_padding_bits :]

        return decoded_bits
