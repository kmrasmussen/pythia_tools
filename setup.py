from setuptools import setup, find_packages

setup(
    name="pythia-tools",
    version="0.1.0",
    description="Code to help understanding Pythia LLMs",
    author="Kasper Rasmussen",
    author_email="kasmura@gmail.com",
    url="https://github.com/kmrasmussen/pythia_tools",
    packages=find_packages(),
    install_requires=['transformers', 'seaborn', 'leidenalg', 'numpy', 'scikit-learn', 'matplotlib', 'tqdm',  'scipy', 'umap-learn', 'statsmodels', 'pandas', 'zstandard'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
)
