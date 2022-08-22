import numpy as np

from sympy import symbols
# from sympy import diff, log, exp, sqrt

from scipy.stats import norm, multivariate_normal

from ._utils_copulas import opt_and_lambdify, read_and_lambdify_sympy_expr
# from ._utils_copulas import sym_copula_derivs_one_par, write_sympy_expr, copula_derivs_one_par

# u_sym, v_sym, par_sym = symbols('u v par')

# gumbel_cdf_sym = exp(-((-log(u_sym))**par_sym + (-log(v_sym))**par_sym)**(1/par_sym))
# gumbel_cop_funs = copula_derivs_one_par(gumbel_cdf_sym, u_sym, v_sym, par_sym)
# gumbel_sym_dict = sym_copula_derivs_one_par(gumbel_cdf_sym, u_sym, v_sym, par_sym)
# write_sympy_expr(gumbel_sym_dict, './vineknockoffs/sym_copula_expr/gumbel.csv')

# gumbel_cop_funs = read_and_lambdify_sympy_expr('vineknockoffs.sym_copula_expr', 'gumbel.csv',
#                                                (par_sym, u_sym, v_sym))

# clayton_cdf_sym = (u_sym**(-par_sym) + v_sym**(-par_sym) - 1)**(-1/par_sym)
# clayton_cop_funs = copula_derivs_one_par(clayton_cdf_sym, u_sym, v_sym, par_sym)
# clayton_sym_dict = sym_copula_derivs_one_par(clayton_cdf_sym, u_sym, v_sym, par_sym)
# write_sympy_expr(clayton_sym_dict, './vineknockoffs/sym_copula_expr/clayton.csv')

# clayton_cop_funs = read_and_lambdify_sympy_expr('vineknockoffs.sym_copula_expr', 'clayton.csv',
#                                                 (par_sym, u_sym, v_sym))

# frank_cdf_sym = - 1/par_sym * \
#                 log(1/(1 - exp(-par_sym)) *
#                     (1 - exp(-par_sym) - (1 - exp(-par_sym*u_sym)) * (1 - exp(-par_sym*v_sym))))
# frank_cop_funs = copula_derivs_one_par(frank_cdf_sym, u_sym, v_sym, par_sym)
# frank_sym_dict = sym_copula_derivs_one_par(frank_cdf_sym, u_sym, v_sym, par_sym)
# write_sympy_expr(frank_sym_dict, './vineknockoffs/sym_copula_expr/frank.csv')

# frank_cop_funs = read_and_lambdify_sympy_expr('vineknockoffs.sym_copula_expr', 'frank.csv',
#                                               (par_sym, u_sym, v_sym))

# x_sym, y_sym, par_sym = symbols('x y par')

# gauss_cop_xy_funs = dict()
# gauss_sym_dict = dict()
# gauss_pdf_sym = 1/(sqrt(1-par_sym**2)) * exp(-(par_sym**2*(x_sym**2 + y_sym**2) - 2*par_sym*x_sym*y_sym)
#                                                / (2*(1-par_sym**2)))
# gauss_sym_dict['pdf'], gauss_cop_xy_funs['pdf'] = opt_and_lambdify(gauss_pdf_sym, x_sym, y_sym, par_sym)
#
# gauss_ll_sym = log(1/(sqrt(1-par_sym**2))) - (par_sym**2*(x_sym**2 + y_sym**2)
#                                                 - 2*par_sym*x_sym*y_sym) / (2*(1-par_sym**2))
# gauss_sym_dict['ll'], gauss_cop_xy_funs['ll'] = opt_and_lambdify(gauss_ll_sym, x_sym, y_sym, par_sym)
#
# gauss_d_ll_d_par_sym = diff(gauss_ll_sym, par_sym)
# gauss_sym_dict['d_ll_d_par'], gauss_cop_xy_funs['d_ll_d_par'] = opt_and_lambdify(gauss_d_ll_d_par_sym,
#                                                                                      x_sym, y_sym, par_sym)
# write_sympy_expr(gauss_sym_dict, './vineknockoffs/sym_copula_expr/gaussian.csv')

# gauss_cop_xy_funs = read_and_lambdify_sympy_expr('vineknockoffs.sym_copula_expr', 'gaussian.csv',
#                                                  (par_sym, x_sym, y_sym))
