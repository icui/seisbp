import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="SeisBP",
    version="0.0.1",
    author="Congyue Cui",
    author_email="ccui@princeton.edu",
    description="A simple library to read and write seismic data using ADIOS2 binary pack format.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/icui/seisbp",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)