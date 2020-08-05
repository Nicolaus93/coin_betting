"""
Created on Tue Feb 11 13:47:02 2020

@author: Zhenxun Zhuang
"""

from setuptools import setup, find_packages

setup(
    name="optimal_pytorch",
    version="0.1",
	description=('PyTorch library for a bunch of optimization methods.'),
	author='Boston University - Optimal Lab',
	url='https://sites.google.com/view/optimal-lab/home',
	license='Apache 2.0',
    packages=find_packages(),
    install_requires=[
        'future',
        'tqdm',
        'numpy',
        'argparse',
        'yapf',
        'torch>=1.0',
        'torchvision>=0.2.2',
        'matplotlib>=1.5.2',
        'line_profiler'
    ],
    classifiers=[
        'Development Status :: 2 - Pre-alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Optimization',
    ],
    keywords='pytorch optimization',
)
