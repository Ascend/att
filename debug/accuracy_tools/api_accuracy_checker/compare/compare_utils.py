from api_accuracy_checker.common.utils import Const, print_warn_log
import numpy as np


class CompareConst:
    NAN = np.nan
    NA = "N/A"


def check_dtype_comparable(x, y):
    dtype_set = {x.dtype, y.dtype}
    if dtype_set.issubset(Const.FLOAT_TYPE) or dtype_set.issubset(Const.BOOL_TYPE) or dtype_set.issubset(Const.INT_TYPE):
        return True
    print_warn_log(f"Compare: Unexpected dtype {x.dtype}, {y.dtype}")
    return False