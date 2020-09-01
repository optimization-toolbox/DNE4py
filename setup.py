import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="DNE4py",
    version="0.0.2",
    author="Hugo Dovs",
    author_email="hugodovs@gmail.com",
    description="DNE4py: Deep Neuroevolution Algorithms using mpi4py",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/optimization-toolbox/DNE4py",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: GNU General Public License v3.0",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
)
