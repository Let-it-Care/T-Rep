from setuptools import setup

def load_requirements(filename):
    with open(filename) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="trep",
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    description="T-Rep: Representation Learning for Time-Series Using Time-Embeddings",
    url="https://github.com/let-it-care/t-rep",
    packages=[".", "models", "tasks"],
    install_requires=load_requirements('requirements.txt'),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.10,<3.11",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)
