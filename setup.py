from setuptools import setup, find_packages

setup(
    name='pytm',
    version='0.1.0',  # Replace with your desired version
    author='Hemeng Wang',
    author_email='wanghemeng@163.com',
    description='A package for managing tensors',
    packages=find_packages(),
    install_requires=[
        'torch',  # Add any other required dependencies
    ]
)
