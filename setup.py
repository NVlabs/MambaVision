""" Setup
"""
from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='mambavision',
    version='1.2.0',
    description='MambaVision: A Hybrid Mamba-Transformer Vision Backbone',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/NVlabs/MambaVision',
    author='Ali Hatamizadeh',
    author_email='ahatamiz123@gmail.com',
    classifiers=[
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],

    # Note that this is a string of words separated by whitespace, not a list.
    keywords='pytorch pretrained models mamba vision transformer vit',
    packages=find_packages(exclude=['tests']),
    include_package_data=True,
    install_requires=['timm==1.0.15', 'transformers==4.50.0', 'mamba-ssm==2.2.4', 'einops==0.8.1', 'requests==2.32.3', 'Pillow==11.1.0'],
    license="NVIDIA Source Code License-NC",
    python_requires='>=3.9',
)
