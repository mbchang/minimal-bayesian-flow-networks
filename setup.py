from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="bfn",
    version="0.1.0",
    author="Michael Chang",
    author_email="mbchang2017@gmail.com",
    description="A minimal reproduction of the basic figures of the Bayesian Flow Networks paper for building intuition.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mbchang/bayesian_flow_networks",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[],
    python_requires=">=3.9",
)
