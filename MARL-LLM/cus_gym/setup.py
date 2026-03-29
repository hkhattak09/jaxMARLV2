from setuptools import setup, find_packages

setup(
    name='gym',
    version='0.19.0',
    description='A customized gym package for MARL-LLM (JAX path)',
    author='Hassan',
    author_email='25100013@lums.edu.pk',
    packages=find_packages(exclude=['tests', 'docs']),
    install_requires=[
        # List your package's dependencies here
        # e.g., 'numpy>=1.19.2', 'requests>=2.25.1'
    ],
    extras_require={
        'dev': [
            # Development dependencies
            'pytest>=6.0.0',  # For testing
            'flake8>=3.8.4',  # For linting
        ],
    },
    entry_points={
        'console_scripts': [
            # Define command-line scripts here
            # 'script_name=your_package.module:function',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # Replace with your license if different
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.12',
)

