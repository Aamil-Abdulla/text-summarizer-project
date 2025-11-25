from setuptools import find_packages, setup

def get_requirements():
    with open("requirements.txt") as f:
        return f.read().splitlines()

setup(
    name="text_summarizer",
    version="0.0.1",
    author="your_name",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=get_requirements(),
)
