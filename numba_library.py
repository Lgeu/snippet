# 参考
# https://ikatakos.com/pot/programming/python/packages/numba

import numpy as np


# >>> numba compile >>>
def numba_compile(numba_config):
    import os, sys
    if sys.argv[-1] == "ONLINE_JUDGE":
        from numba import njit
        from numba.pycc import CC
        cc = CC("my_module")
        for func, signature in numba_config:
            vars()[func.__name__] = njit(signature)(func)
            cc.export(func.__name__, signature)(func)
        cc.compile()
        exit()
    elif os.name == "posix":
        exec(f"from my_module import {','.join(func.__name__ for func, _ in numba_config)}")
        for func, _ in numba_config:
            globals()[func.__name__] = vars()[func.__name__]
    else:
        from numba import njit
        for func, signature in numba_config:
            globals()[func.__name__] = njit(signature, cache=True)(func)
        print("compiled!", file=sys.stderr)
numba_compile([
    [solve, "i8(i8,i8,i8[:,:])"],
])
# <<< numba compile <<<


# >>> binary indexed tree >>>
# 必要な要素数+1 の長さの ndarray の 1 要素目以降を使う
def bit_sum(bit, i):  # [bit_sum, "i8(i8[:],i8)"],
    # (0, i]
    res = 0
    while i:
        res += bit[i]
        i -= i & -i
    return res
def bit_add(bit, i, val):  # [bit_add, "void(i8[:],i8,i8)"],
    n = len(bit)
    while i < n:
        bit[i] += val
        i += i & -i
# <<< binary indexed tree <<<


def inversion_number(arr):  # [inversion_number, "i8(f8[:])"],
    # 転倒数
    n = len(arr)
    arr = np.argsort(arr) + 1
    bit = np.zeros(n+1, dtype=np.int64)
    res = n * (n-1) >> 1
    for val in arr:
        res -= bit_sum(bit, val)
        bit_add(bit, val, 1)
    return res


def numba_pow(base, exp, mod):  # [numba_pow, "i8(i8,i8,i8)"],
    exp %= mod - 1
    res = 1
    while exp:
        if exp % 2:
            res = res * base % mod
        base = base * base % mod
        exp //= 2
    return res


def z_algo(S):  # [z_algo, "i8[:](i8[:])"],
    # Z-algoirhm  O(n)
    # Z[i] := S と S[i:] で prefix が何文字一致しているか
    # 検証: https://atcoder.jp/contests/abc150/submissions/15829530
    i, j, n = 1, 0, len(S)
    Z = np.zeros_like(S)
    Z[0] = n
    while i < n:
        while i+j < n and S[j] == S[i+j]:
            j += 1
        if j == 0:
            i += 1
            continue
        Z[i] = j
        d = 1
        while i+d < n and d+Z[d] < j:
            Z[i+d] = Z[d]
            d += 1
        i += d
        j -= d
    return Z
