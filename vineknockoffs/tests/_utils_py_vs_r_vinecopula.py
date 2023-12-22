import pytest

rpy2 = pytest.importorskip("rpy2")
from rpy2 import robjects


def py_copula_funs_eval(data, cop_obj, fun_type):
    if fun_type == 'cdf':
        res = cop_obj.cdf(data[:, 0], data[:, 1])
    elif fun_type == 'pdf':
        res = cop_obj.pdf(data[:, 0], data[:, 1])
    elif fun_type == 'hfun':
        res = cop_obj.hfun(data[:, 0], data[:, 1])
    elif fun_type == 'vfun':
        res = cop_obj.vfun(data[:, 0], data[:, 1])
    elif fun_type == 'd_hfun_d_par':
        res = cop_obj.d_hfun_d_par(data[:, 0], data[:, 1])
    elif fun_type == 'd_vfun_d_par':
        res = cop_obj.d_vfun_d_par(data[:, 0], data[:, 1])
    elif fun_type == 'd_hfun_d_v':
        res = cop_obj.d_hfun_d_v(data[:, 0], data[:, 1])
    else:
        assert fun_type == 'd_vfun_d_u'
        res = cop_obj.d_vfun_d_u(data[:, 0], data[:, 1])

    return res


r_copula_funs_eval = robjects.r('''
        library(VineCopula)
        function(u, v, family, par, type) {

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
          else if (type == 'd_hfun_d_par'){
            res = BiCopHfuncDeriv(u, v, family, par, deriv="par")
          }
          else if (type == 'd_vfun_d_par'){
            # flip arguments to get vfun (only valid for symmetric copulas)
            res = BiCopHfuncDeriv(v, u, family, par, deriv="par")
          }
          else if (type == 'd_hfun_d_v'){
            res = BiCopHfuncDeriv(u, v, family, par, deriv="u2")
          }
          else if (type == 'd_vfun_d_u'){
            # flip arguments to get vfun (only valid for symmetric copulas)
            res = BiCopHfuncDeriv(v, u, family, par, deriv="u2")
          }
          return(res)
        }
        ''')
