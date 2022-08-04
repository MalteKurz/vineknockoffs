from setuptools import setup

with open('README.md', 'r') as fh:
    long_description = fh.read()

# PROJECT_URLS = {
#     'Bug Tracker': 'https://github.com/MalteKurz/vineknockoffs/issues',
#     'Source Code': 'https://github.com/MalteKurz/vineknockoffs'
# }

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
    extras_require={
        'KDE1D': ['rpy2'],
        # 'KDE1D': ['pykde1d @ git+https://github.com/vinecopulib/pykde1d.git@main#egg=pykde1d'],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
