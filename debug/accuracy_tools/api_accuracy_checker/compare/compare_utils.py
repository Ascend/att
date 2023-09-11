from api_accuracy_checker.common.utils import Const, print_warn_log
import numpy as np


class CompareConst:
    NAN = np.nan
    NA = "N/A"


def check_dtype_comparable(x, y):
    if x.dtype in Const.FLOAT_TYPE and y.dtype in Const.FLOAT_TYPE:
        return False
    if x.dtype in Const.BOOL_TYPE and y.dtype in Const.BOOL_TYPE:
        return False
    if x.dtype in Const.INT_TYPE and y.dtype in Const.INT_TYPE:
        return False
    print_warn_log(f"Compare: Unexpected dtype {x.dtype}, {y.dtype}")
    return True