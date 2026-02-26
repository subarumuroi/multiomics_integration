from setuptools import setup, find_packages

setup(
    name='multiomics_integration',
    version='0.3.0',
    author='Subaru Muroi',
    author_email='k.muroi@uq.edu.au',
    description='Multi-omics integration: sPLS-DA/DIABLO, Random Forest, Ordinal Regression, WGCNA.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    py_modules=['ingestion', 'visualization', 'utils'],
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'mord',
        'shap',
        'scipy',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)