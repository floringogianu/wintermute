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
    install_requires=["gym[atari]", "lycon==0.1.9", "rl_logger==0.1"],
    dependency_links=[
        "https://github.com/floringogianu/rl_logger/tarball/master#egg=rl_logger-0.1"
    ],
    zip_safe=False,
)
