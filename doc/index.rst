vineknockoffs: Vine copula based knockoffs
==========================================

.. |Unit tests| image:: https://github.com/MalteKurz/vineknockoffs/actions/workflows/unitest.yml/badge.svg
   :target: https://github.com/MalteKurz/vineknockoffs/actions/workflows/unitest.yml
.. |PyPI version| image:: https://badge.fury.io/py/vineknockoffs.svg
   :target: https://badge.fury.io/py/vineknockoffs
.. |codecov| image:: https://codecov.io/gh/MalteKurz/vineknockoffs/branch/main/graph/badge.svg?token=E3O3ZOLLBE
   :target: https://codecov.io/gh/MalteKurz/vineknockoffs
.. |Python version| image:: https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9%20%7C%203.10-blue
   :target: https://www.python.org


The repo `<https://github.com/MalteKurz/vineknockoffs>`_ contains an implementation of vine copula knockoffs for
high-dimensional controlled variable selection, see
`Kurz (2022) <https://arxiv.org/abs/2210.11196>`_ for details. The documentation is work in progress.

## Main features

The python package vineknockoffs can be used to estimate (for details see `Kurz (2022) <https://arxiv.org/abs/2210.11196>`_)
- Gaussian knockoff models,
- Gaussian copula knockoffs models,
- Vine copula knockoff models.


## Citation

If you use the vineknockoffs package a citation is highly appreciated:

Kurz, M. S. (2022). Vine copula based knockoff generation for high-dimensional controlled variable selection,
arXiv:`2210.11196<https://arxiv.org/abs/2210.11196>`_.

.. code-block:: TeX

    @misc{Kurz2022vineknockoffs,
          title={Vine copula based knockoff generation for high-dimensional controlled variable selection},
          author={M. S. Kurz},
          year={2022},
          eprint={2210.11196},
          archivePrefix={arXiv},
          primaryClass={stat.ME},
          note={arXiv:\href{https://arxiv.org/abs/2210.11196}{2210.11196} [stat.ME]}
    }

## Acknowledgements

Funding by the Deutsche Forschungsgemeinschaft (DFG, German Research
Foundation) is acknowledged – Project Number 431701914.

## References

Kurz, M. S. (2022). Vine copula based knockoff generation for high-dimensional controlled variable selection,
arXiv:`2210.11196<https://arxiv.org/abs/2210.11196>`_.

.. toctree::
   :hidden:

   intro
   api
