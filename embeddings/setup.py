from setuptools import find_packages, setup

try:
    long_description = open("README.md").read()
except FileNotFoundError:
    long_description = ""

setup(
    name="robust_steganography",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "openai",
        "torch",
        "sentence-transformers",
        "tqdm",
        "scikit-learn",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "flake8",
        ],
    },
    author="Neil Perry",
    author_email="naperry@stanford.edu",
    description="Embeddings-based steganography system for hiding messages in text",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NeilAPerry/robust_steganography",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
