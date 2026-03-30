from setuptools import setup, find_packages

setup(
    name='cus-gym',
    version='1.0.0',
    description='Custom lightweight gym for MARL-LLM with JAX integration (no legacy dependencies)',
    author='Hassan',
    author_email='25100013@lums.edu.pk',
    packages=find_packages(exclude=['tests', 'docs']),
    install_requires=[
        'numpy>=1.24.0',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.12',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.12',
)

