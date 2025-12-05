from setuptools import setup, find_packages

setup(
    name="medical-ai-system",
    version="2.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "fastapi",
        "uvicorn",
        "pydantic",
        "transformers",
        "torch",
        "langchain",
        "chromadb",
        "sentence-transformers",
        "pandas",
        "numpy",
    ],
)
