from setuptools import setup, find_packages

setup(
    name="wintermute",
    version="0.1.0",
    description="A library of reinforcement learning primitives.",
    url="https://github.com/floringogianu/wintermute",
    author="Florin Gogianu",
    author_email="florin.gogianu@gmail.com",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "gym[atari]>=0.9.6",
        "torch>=1.0.0"
    ],
    zip_safe=False,
)
