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
            globals()[func.__name__] = njit(signature)(func)
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
def bitify(arr):  # [bitify, "void(i8[:])"],
    # len(arr) は 2 冪 + 1
    for i in range(1, len(arr)-1):
        arr[i + (i & -i)] += arr[i]
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


def pow_mod(base, exp):  # [numba_pow, "i8(i8,i8)"],
    # mod はグローバル変数を参照
    exp %= mod - 1
    res = 1
    while exp:
        if exp % 2:
            res = res * base % mod
        base = base * base % mod
        exp //= 2
    return res

def comb_cunstruct(n):  # [comb_cunstruct, "Tuple((i8[:],i8[:]))(i8,)"],
    # mod はグローバル変数を参照
    fac = np.empty(n + 1, dtype=np.int64)
    facinv = np.empty(n + 1, dtype=np.int64)
    fac[0] = f = 1
    for i in range(1, n + 1):
        f = f * i % mod
        fac[i] = f
    f = pow_mod(f, -1)
    for i in range(n, -1, -1):
        facinv[i] = f
        f = f * i % mod
    return fac, facinv

def comb(n, r, fac, facinv):  # [comb, "i8(i8,i8,i8[:],i8[:])"],
    # mod はグローバル変数を参照
    return fac[n] * facinv[r] % mod * facinv[n - r] % mod


def z_algo(S):  # [z_algo, "i8[:](i8[:])"],
    # Z-algoirhm  O(n)
    # Z[i] := S と S[i:] で prefix が何文字一致しているか
    # 検証1: https://atcoder.jp/contests/abc150/submissions/15829530
    # 検証2: https://atcoder.jp/contests/abc141/submissions/15855247
    i, j, n = 1, 0, len(S)
    Z = np.zeros(S.shape, dtype=np.int64)
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


def sort_edges(N, edges_):  # [sort_edges, "Tuple((i8[:],i8[:]))(i8,i8[:,:])"],
    # N: 頂点番号の最大値
    M = len(edges_)
    edges = np.empty((M * 2, 2), dtype=np.int64)
    edges[:M] = edges_
    edges[M:] = edges_[:, ::-1]
    order = np.argsort(edges[:, 0])  # O(N) にできなくもない
    edges = edges[order, 1]
    c = np.zeros(N+1, dtype=np.int64)
    c_ = np.bincount(edges_.ravel())  # minlength を使わせて
    c[:len(c_)] = c_
    c = np.cumsum(c)
    lefts = np.zeros(len(c) + 1, dtype=np.int64)
    lefts[1:] = c
    return edges, lefts

def eular_tour(edges, lefts, root):  # [eular_tour, "Tuple((i8[:],i8[:],i8[:],i8[:]))(i8[:],i8[:],i8)"],
    # グラフは 1-indexed が良い
    n = len(lefts)-1
    stack = [root]
    tour = [0] * 0
    firsts = np.full(n, -100, dtype=np.int64)
    lasts = np.full(n, -100, dtype=np.int64)
    parents = np.full(n, -100, dtype=np.int64)
    while stack:
        v = stack.pop()
        if firsts[v] >= 0:
            lasts[v] = len(tour)
            tour.append(-v)  # 帰りがけの辺の表現をマイナス以外にしたい場合ここを変える
            continue
        p = parents[v]
        firsts[v] = len(tour)
        tour.append(v)
        stack.append(v)
        for u in edges[lefts[v]:lefts[v+1]]:
            if p != u:
                parents[u] = v
                stack.append(u)
    tour = np.array(tour, dtype=np.int64)
    return tour, firsts, lasts, parents


from functools import reduce
def rerooting(n, edges):  # [rerooting, "(i8,i8[:,:])"],
    # 全方位木 dp
    # 参考1: https://qiita.com/keymoon/items/2a52f1b0fb7ef67fb89e
    # 参考2: https://atcoder.jp/contests/abc160/submissions/15255726
    # 検証: https://atcoder.jp/contests/abc160/submissions/15971370

    # >>> ここを変える >>>
    # 必要な情報は引数に持たせる
    identity = (1, 0)
    def merge(a, b):
        return a[0] * b[0] % mod * comb(a[1] + b[1], a[1], fac, facinv) % mod, a[1] + b[1]
    def add_node(value, idx):
        return value[0], value[1] + 1
    # <<< ここを変える <<<

    G = [[0]*0 for _ in range(n)]
    for i in range(n-1):
        a, b = edges[i]
        G[a].append(b)
        G[b].append(a)
    # step 1
    order = []  # 行きがけ順
    stack = [0]
    while stack:
        v = stack.pop()
        order.append(v)
        for u in G[v]:
            stack.append(u)
            G[u].remove(v)
    # 下から登る
    dp_down = [identity] * n  # 自身とその下
    for v in order[:0:-1]:
        dp_down[v] = add_node(reduce(
            merge, [dp_down[u] for u in G[v]], identity
        ), v)
    # step 2
    # 上から降りる
    dp_up = [identity] * n  # 親とその先
    for v in order:
        Gv = G[v]
        if len(Gv) == 0:
            continue
        cum = identity
        right = [identity]
        for u in Gv[:0:-1]:
            cum = merge(dp_down[u], cum)
            right.append(cum)
        right.reverse()
        cum = dp_up[v]
        for u, cum_r in zip(Gv, right):
            dp_up[u] = add_node(merge(cum, cum_r), v)
            cum = merge(cum, dp_down[u])
    results = [identity] * 0
    for v, Gv in enumerate(G):
        results.append(add_node(
            reduce(merge, [dp_down[u] for u in Gv], dp_up[v]), v
        ))
    return np.array(results)


# セグメント木: https://atcoder.jp/contests/abc158/submissions/16233600
# 平方分割（遅延評価）: https://atcoder.jp/contests/abc177/submissions/16376895

