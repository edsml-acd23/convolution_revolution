from setuptools import setup, find_packages

setup(
    name='wiener_transformer',
    version='0.1',
    author='Andrei Danila',
    packages=find_packages(include=['wiener_transformer', 'wiener_attention']),
)