from setuptools import setup, find_packages

with open('README.md', 'r') as fh:
    long_description = fh.read()

#PROJECT_URLS = {
#    'Bug Tracker': 'https://github.com/MalteKurz/vineknockoffs/issues',
#    'Source Code': 'https://github.com/MalteKurz/vineknockoffs'
#}

setup(
    name='vineknockoffs',
    version='0.0.dev0',
    author='Kurz, M. S.',
    maintainer='Malte S. Kurz',
    maintainer_email='malte.kurz@tum.de',
    description=' Vine copula based knockoffs ',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/MalteKurz/vineknockoffs',
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'statsmodels',
        'sympy',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
