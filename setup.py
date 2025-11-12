"""
Setup configuration for FSO Channel Estimation package.

This package provides Free-Space Optical communication channel estimation
using machine learning techniques.
"""

from setuptools import setup, find_packages

setup(
    name='fso-channel-estimation',
    version='0.1.0',
    description='Free-Space Optical communication channel estimation using machine learning',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='TBD',
    author_email='TBD',
    url='TBD',
    packages=find_packages(),
    python_requires='>=3.10',
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'pandas',
        'scikit-learn',
        'xgboost',
        'catboost',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Physics',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    keywords='fso optical-communication channel-estimation machine-learning',
)
