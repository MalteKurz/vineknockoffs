from sympy.utilities.lambdify import lambdify
from sympy.printing.numpy import SciPyPrinter
from sympy.codegen.rewriting import optimize, optims_c99
from sympy import diff, log
import csv
from importlib.resources import open_text


def opt_and_lambdify(fun, u_sym, v_sym, theta_sym):
    fun = optimize(fun, optims_c99)
    ufun = lambdify((theta_sym, u_sym, v_sym),
                    fun,
                    'numpy')
    return fun, ufun


def opt_and_print(fun):
    fun = optimize(fun, optims_c99)
    fun_str = SciPyPrinter().doprint(fun)
    return fun_str


def copula_derivs_one_par(cdf_sym, u_sym, v_sym, theta_sym):
    ufuns = dict()

    cdf_sym, ufuns['cdf'] = opt_and_lambdify(cdf_sym,
                                             u_sym, v_sym, theta_sym)

    hfun_sym = diff(cdf_sym, v_sym)
    hfun_sym, ufuns['hfun'] = opt_and_lambdify(hfun_sym,
                                               u_sym, v_sym, theta_sym)

    vfun_sym = diff(cdf_sym, u_sym)
    vfun_sym, ufuns['vfun'] = opt_and_lambdify(vfun_sym,
                                               u_sym, v_sym, theta_sym)

    pdf_sym = diff(hfun_sym, u_sym)
    pdf_sym, ufuns['pdf'] = opt_and_lambdify(pdf_sym,
                                             u_sym, v_sym, theta_sym)

    ll_sym = log(pdf_sym)
    ll_sym, ufuns['ll'] = opt_and_lambdify(ll_sym,
                                           u_sym, v_sym, theta_sym)

    d_ll_d_theta_sym = diff(ll_sym, theta_sym)
    d_ll_d_theta_sym, ufuns['d_ll_d_theta'] = opt_and_lambdify(d_ll_d_theta_sym,
                                                               u_sym, v_sym, theta_sym)

    d_hfun_d_theta_sym = diff(hfun_sym, theta_sym)
    d_hfun_d_theta_sym, ufuns['d_hfun_d_theta'] = opt_and_lambdify(d_hfun_d_theta_sym,
                                                                   u_sym, v_sym, theta_sym)

    d_vfun_d_theta_sym = diff(vfun_sym, theta_sym)
    d_vfun_d_theta_sym, ufuns['d_vfun_d_theta'] = opt_and_lambdify(d_vfun_d_theta_sym,
                                                                   u_sym, v_sym, theta_sym)

    d_hfun_d_v_sym = diff(hfun_sym, v_sym)
    d_hfun_d_v_sym, ufuns['d_hfun_d_v'] = opt_and_lambdify(d_hfun_d_v_sym,
                                                           u_sym, v_sym, theta_sym)

    d_vfun_d_u_sym = diff(vfun_sym, u_sym)
    d_vfun_d_u_sym, ufuns['d_vfun_d_u'] = opt_and_lambdify(d_vfun_d_u_sym,
                                                           u_sym, v_sym, theta_sym)

    return ufuns


def sym_copula_derivs_one_par(cdf_sym, u_sym, v_sym, theta_sym):
    sym_funs = dict()

    sym_funs['cdf'] = opt_and_print(cdf_sym)

    hfun_sym = diff(cdf_sym, v_sym)
    sym_funs['hfun'] = opt_and_print(hfun_sym)

    vfun_sym = diff(cdf_sym, u_sym)
    sym_funs['vfun'] = opt_and_print(vfun_sym)

    pdf_sym = diff(hfun_sym, u_sym)
    sym_funs['pdf'] = opt_and_print(pdf_sym)

    ll_sym = log(pdf_sym)
    sym_funs['ll'] = opt_and_print(ll_sym)

    d_ll_d_theta_sym = diff(ll_sym, theta_sym)
    sym_funs['d_ll_d_theta'] = opt_and_print(d_ll_d_theta_sym)

    d_hfun_d_theta_sym = diff(hfun_sym, theta_sym)
    sym_funs['d_hfun_d_theta'] = opt_and_print(d_hfun_d_theta_sym)

    d_vfun_d_theta_sym = diff(vfun_sym, theta_sym)
    sym_funs['d_vfun_d_theta'] = opt_and_print(d_vfun_d_theta_sym)

    d_hfun_d_v_sym = diff(hfun_sym, v_sym)
    sym_funs['d_hfun_d_v'] = opt_and_print(d_hfun_d_v_sym)

    d_vfun_d_u_sym = diff(vfun_sym, u_sym)
    sym_funs['d_vfun_d_u'] = opt_and_print(d_vfun_d_u_sym)

    return sym_funs


def write_sympy_expr(sym_dict, filename):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['fun_name', 'lambdarepr']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for key, value in sym_dict.items():
            writer.writerow({'fun_name': key, 'lambdarepr': value})
    return


def read_sympy_expr(filename):
    sym_dict = dict()
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            sym_dict[row['fun_name']] = row['lambdarepr']
    return sym_dict


def read_and_lambdify_sympy_expr(package, resource, args):
    ufuns = dict()
    with open_text(package, resource) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            ufuns[row['fun_name']] = lambdify(args,
                                              row['lambdarepr'],
                                              'numpy')
    return ufuns
