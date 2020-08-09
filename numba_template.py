import os
import sys

numba_config = [
    ["proc", "i4[:](i4,i4[:])"],
]
if sys.argv[-1] == "ONLINE_JUDGE":
    from numba.pycc import CC
    cc = CC("my_module")
    for func_name, types in numba_config:
        cc.export(func_name, types)(vars()[func_name])
        cc.compile()
    exit()
elif os.name == "posix":
    exec(f"from my_module import {','.join(func_name for func_name, _ in numba_config)}")
else:
    from numba import njit
    for func_name, types in numba_config:
        vars()[func_name] = njit(types, cache=True)(vars()[func_name])
    print("compiled", file=sys.stderr)


# 参考
# https://ikatakos.com/pot/programming/python/packages/numba
