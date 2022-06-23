import pytest

rpy2 = pytest.importorskip("rpy2")
from rpy2 import robjects


def py_copula_funs_eval(data, cop_obj, cop_par, fun_type):
    if fun_type == 'cdf':
        res = cop_obj.cdf(cop_par, data[:, 0], data[:, 1])
    elif fun_type == 'pdf':
        res = cop_obj.pdf(cop_par, data[:, 0], data[:, 1])
    elif fun_type == 'hfun':
        res = cop_obj.hfun(cop_par, data[:, 0], data[:, 1])
    elif fun_type == 'vfun':
        res = cop_obj.vfun(cop_par, data[:, 0], data[:, 1])

    return res


r_copula_funs_eval = robjects.r('''
        library(VineCopula)
        copula_funs_eval <- function(u, v, family, par, type) {

          if (type == 'cdf'){
            res = BiCopCDF(u, v, family, par)
          }
          else if (type == 'pdf'){
            res = BiCopPDF(u, v, family, par)
          }
          else if (type == 'hfun'){
            res = BiCopHfunc2(u, v, family, par)
          }
          else if (type == 'vfun'){
            res = BiCopHfunc1(u, v, family, par)
          }
          return(res)
        }
        ''')
