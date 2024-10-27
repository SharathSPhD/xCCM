# Save as: setup.py

from setuptools import setup, find_packages

setup(
    name="xccm",
    version="0.2.0",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.18.0',
        'scipy>=1.4.0',
        'scikit-learn>=0.22.0',
        'pandas>=1.0.0'
    ],
    extras_require={
        'test': [
            'pytest>=6.0.0',
            'pytest-cov>=2.10.0'
        ],
        'dev': [
            'black>=20.8b1',
            'flake8>=3.8.0',
            'mypy>=0.800'
        ]
    },
    python_requires='>=3.7',
    author="xCCM Contributors",
    description="Extended Convergent Cross Mapping implementation",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)