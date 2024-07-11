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
    name='fastervit',
    version='0.9.8',
    description='FasterViT: Fast Vision Transformers with Hierarchical Attention',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/NVlabs/FasterViT',
    author='Ali Hatamizadeh',
    author_email='ahatamiz123@gmail.com',
    classifiers=[
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],

    # Note that this is a string of words separated by whitespace, not a list.
    keywords='pytorch pretrained models efficientnet mobilenetv3 mnasnet resnet vision transformer vit',
    packages=find_packages(exclude=['tests']),
    include_package_data=True,
    install_requires=['timm >= 0.6.12', 'torchvision', 'pyyaml'],
    license="NVIDIA Source Code License-NC",
    python_requires='>=3.7',
)
