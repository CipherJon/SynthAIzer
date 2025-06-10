from setuptools import setup, find_packages

setup(
    name="synthaizer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.6.0,<3.0.0",  # Core ML dependency
        "numpy>=1.22.0,<2.0.0",  # Core numerical dependency
        "pretty_midi>=0.2.10,<1.0.0",  # Core MIDI handling
        "python-rtmidi>=1.5.0,<2.0.0",  # Core MIDI I/O
        "mido>=1.2.10,<2.0.0",  # Core MIDI file handling
        "requests>=2.32.3,<3.0.0",  # Core HTTP client
        "python-dotenv>=1.0.0,<2.0.0",  # Core environment handling
    ],
    extras_require={
        "dev": [
            "scipy>=1.7.0,<2.0.0",  # Optional scientific computing
            "tqdm>=4.65.0,<5.0.0",  # Optional progress bars
        ],
        "full": [
            "scipy>=1.7.0,<2.0.0",
            "tqdm>=4.65.0,<5.0.0",
        ]
    },
    python_requires=">=3.8",
    author="SynthAIzer Team",
    author_email="your.email@example.com",
    description="AI-powered music generator for LMMS",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/synthaizer",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
) 