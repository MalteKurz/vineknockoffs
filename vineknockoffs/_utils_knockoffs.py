import numpy as np
try:
    import rpy2
except ImportError:
    raise ImportError()
from rpy2 import robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
r_kde1d_available = robjects.r('library("glmnet", quietly=TRUE, logical.return=TRUE)')[0]
if not r_kde1d_available:
    raise ImportError()


cv_glmnet_r = robjects.r('''
        cv_glmnet <- function(x, y, s) {
          fit <- glmnet::cv.glmnet(x, y)
          return(as.numeric(coef(fit, s=s)))
        }
        ''')
