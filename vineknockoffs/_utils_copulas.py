from sympy.utilities.lambdify import lambdify
from sympy.codegen.rewriting import optimize, optims_c99
from sympy import diff, log


def opt_and_lambdify(fun, u_sym, v_sym, theta_sym):
    fun = optimize(fun, optims_c99)
    ufun = lambdify((theta_sym, u_sym, v_sym),
                    fun,
                    'numpy')
    return fun, ufun


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
