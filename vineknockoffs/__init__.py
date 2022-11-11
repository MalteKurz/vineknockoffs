from pkg_resources import get_distribution

__version__ = get_distribution('vineknockoffs').version

from .vine_knockoffs import VineKnockoffs
