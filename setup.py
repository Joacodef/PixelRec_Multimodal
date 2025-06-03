from setuptools import setup, find_packages

setup(
    name="multimodal-recommender",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.2.1",
        "transformers>=4.47.1",
        "pandas>=2.2.2",
        "numpy>=1.26.4",
        "scikit-learn>=1.6.1",
        "pillow>=10.2.0",
        "tqdm>=4.67.1",
        "pyyaml>=6.0",
        "matplotlib>=3.10.3",
    ],
    python_requires=">=3.7",
)