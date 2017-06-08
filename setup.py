import os
from setuptools import setup, find_packages
from io import open

packages = ['elfie'] + ['elfie.' + p for p in find_packages('elfie')]

with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

setup(
    name='elfie',
    packages=packages,
    version=0.1,
    author='Antti Kangasrääsiö',
    author_email='antti.kangasraasio@iki.fi',
    url='https://github.com/akangasr/elfie',
    install_requires=requirements,
    description='ELFI Experiment framework',
    license='MIT')
