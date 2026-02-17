"""
Setup file for the Lemonade integration module
This file allows installation as a standalone package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("LICENSE", "r", encoding="utf-8") as fh:
    license_content = fh.read()

setup(
    name="lemonade-integration",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A clean interface for interacting with the Lemonade LLM server",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/lemonade-integration",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "requests>=2.25.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "flake8",
        ],
    },
    keywords="llm, ai, lemonade, integration, api",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/lemonade-integration/issues",
        "Source": "https://github.com/yourusername/lemonade-integration",
    },
)