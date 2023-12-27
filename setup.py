from setuptools import setup, find_packages

with open('README.md', 'r') as fh:
    long_description = fh.read()

# PROJECT_URLS = {
#     'Bug Tracker': 'https://github.com/MalteKurz/vineknockoffs/issues',
#     'Source Code': 'https://github.com/MalteKurz/vineknockoffs'
# }

setup(
    name='vineknockoffs',
    version='0.2.dev0',
    author='Kurz, M. S.',
    maintainer='Malte S. Kurz',
    maintainer_email='malte.kurz@tum.de',
    description='Vine copula based knockoffs',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/MalteKurz/vineknockoffs',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'statsmodels',
        'cvxpy',
        'cvxopt',
        'python_tsp'
    ],
    python_requires=">=3.7",
    extras_require={
        'KDE1D': ['rpy2'],
        'SYMPY': ['sympy'],
        # 'KDE1D': ['pykde1d @ git+https://github.com/vinecopulib/pykde1d.git@main#egg=pykde1d'],
        # 'PYCONCORDE': ['pyconcorde @ git+https://github.com/jvkersch/pyconcorde.git@master#egg=pyconcorde'],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
