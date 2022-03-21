from setuptools import setup, find_packages

setup(
    name="cams",
    version="0.1",
    packages=find_packages(exclude=["tests*"]),
    license="MIT",
    description="CAMSStacker",
    # long_description=open("README.md").read(),
    install_requires=[],
    author="Aria Ghora Prabono",
    author_email="hello@ghora.net",
)
