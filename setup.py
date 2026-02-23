from setuptools import setup, find_packages

setup(
    name='multiomics_integration',
    version='0.1.0',
    author='Subaru Muroi',
    author_email='k.muroi@uq.edu.au',
    description='Consolidated multi-omics classification: sPLS-DA/DIABLO, Random Forest, Ordinal Regression.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'mord',
        'shap',
        'umap-learn',
        'scipy',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)