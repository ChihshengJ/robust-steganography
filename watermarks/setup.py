from setuptools import setup, find_packages

setup(
    name="watermark",
    version="0.1",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        "torch",
        "tqdm",
        "pycryptodome",
        "bitstring",
        "numpy",
        "transformers",
        "pynacl",  # for nacl.hash
        "cryptography",  # for HKDF
        "pytest",  # for testing
    ],
    python_requires='>=3.7',
    description="Watermarking library for text generation",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/watermark",
)