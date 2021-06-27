def make_partition_list(N, mod=10**9+7):
    # http://d.hatena.ne.jp/inamori/20121216/p1
    # N 以下の分割数のリストを返す O(n**(3/2))
    # mod は素数でなくても良い
    from itertools import count
    P = [0]*(N+1)
    P[0] = 1
    for n in range(1, N+1):
        p = 0
        m1 = 0
        for k in count(1):
            m1 += 3*k - 2  # m1 = k * (3*k-1) // 2
            if n < m1:
                break
            p += P[n-m1] if k%2==1 else -P[n-m1]
            m2 = m1 + k  # m2 = k * (3*k+1) // 2
            if n < m2:
                break
            p += P[n-m2] if k%2==1 else -P[n-m2]
            p %= mod
        P[n] = p
    return P


import sys
from functools import lru_cache
sys.setrecursionlimit(500000)
mod = 10**9+7
@lru_cache(maxsize=None)
def partition(n, k):  # 自然数 n を k 個の自然数の和で表す場合の数
    if n < 0 or n < k:
        return 0
    elif k == 1 or n == k:
        return 1
    else:
        return (partition(n-k, k) + partition(n-1, k-1)) % mod  # 1 を使わない場合と使う場合の和


"""
# 原始ピタゴラス数
from math import gcd
N = 1500000
cnt = [0]*(N+1)
L = []
for m in range(2, 10**4):
    for n in range(1, m):
        a = m*m-n*n
        b = 2*m*n
        c = m*m+n*n
        if gcd(gcd(a,b),c)!=1:  # 原始ピタゴラス数以外も生成するので弾く
            continue
        L.append(sorted([a,b,c]))
"""

def euler_phi(n):
  # http://tjkendev.github.io/procon-library/python/prime/eulers-totient-function.html
  # オイラーのφ関数
  res = n
  for x in range(2, int(n**.5)+1):
    if n % x == 0:
      res = res // x * (x-1)
      while n % x == 0:
        n //= x
  return res


"""
# オイラーのφ関数（前計算あり）
N = 10**7
isPrime = [True] * (N+1)
isPrime[0] = isPrime[1] = False
for i in range(2, int((N+1)**0.5)+1):
    if isPrime[i]:
        for j in range(i*i, N+1, i):
            isPrime[j] = False
primes = [i for i, f in enumerate(isPrime) if f]
prime_factors = [[] for _ in range(N+1)]
for p in primes:
    for i in range(p, N+1, p):
        prime_factors[i].append(p)
def euler_phi(n):
    for p in prime_factors[n]:
        n = n // p * (p-1)
    return n
"""

"""
# Project Euler 80
# 平方根、任意精度小数
from decimal import Decimal, getcontext
getcontext().prec = 111
ans = 0
for i in range(100):
    s = str(Decimal(i).sqrt()).replace(".", "")
    if len(s)>100:
        ans += sum(map(int, s[:100]))
print(ans)
"""


from math import gcd, sqrt
from collections import defaultdict
from itertools import count
def continued_frac(n):
    # sqrt(n) の連分数展開
    
    sqrt_n = int(sqrt(n))
    if sqrt_n**2 == n:
        return [sqrt_n], 0
    a0 = r = sqrt_n
    
    def solve(right, denom):
        # (sqrt(n) - right) / denom を 1 / (? + (sqrt(n)-?) / ?) にする
        assert right > 0, (n, right, denom)
        denom_new = (n - right*right) // denom  # 必ず割り切れる？？
        a, m = divmod(sqrt_n+right, denom_new)
        return a, sqrt_n-m, denom_new

    dd = defaultdict(lambda: -2)
    d = 1
    dd[(r, d)] = -1
    res = [a0]
    for i in count():
        a, r, d = solve(r, d)
        res.append(a)
        if (r, d) in dd:
            period = i - dd[(r, d)]
            break
        dd[(r, d)] = i
    return res, period

def inv_continued_frac(L):
    # 正則連分数を通常の分数に戻す
    numer, denom = 1, L[-1]
    for v in L[-2::-1]:
        numer += v * denom
        denom, numer = numer, denom
    denom, numer = numer, denom
    return numer, denom

def solve_pell(n):
    # ペル方程式 x*x - n*y*y = 1 の最小整数解
    # 解無しのとき -1, -1 を返す
    if int(sqrt(n))**2 == n:
        return -1, -1
    con_fr, period = continued_frac(n)
    x, y = inv_continued_frac(con_fr[:-1])
    if period%2 == 1:
        x, y = x*x + y*y*n, 2*x*y
    return x, y

def more_pell_solutions(n, x, y):
    # x, y が解のとき、
    #   x_k + y_k √n = (x + y √n)^k
    # もすべて解となる（ブラーマグプタの恒等式）
    yield x, y
    x_, y_ = x, y
    while True:
        x, y = x*x_+n*y*y_, x*y_+y*x_
        yield x, y

def solve_pell_minus(n):
    # ペル方程式の拡張 x*x - n*y*y = -1 の最小整数解
    # 解無しのとき -1, -1 を返す
    if int(sqrt(n))**2 == n:
        return -1, -1
    con_fr, period = continued_frac(n)
    x, y = inv_continued_frac(con_fr[:-1])
    if period%2 == 1:
        return x, y
    else:
        return (-1, -1)
    
def more_pell_minus_solutions(n, x, y):
    x_, y_ = x*x + y*y*n, 2*x*y
    while True:
        x, y = x*x_+n*y*y_, x*y_+y*x_
        yield x, y
    

# カレンダー系
# https://atcoder.jp/contests/arc010/submissions/6917100
# https://atcoder.jp/contests/arc002/submissions/6923018
days = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
days_leap = [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
def is_leap_year(y):  # 閏年判定
    if y%400==0: return True
    elif y%100==0: return False
    elif y%4==0: return True
    else: return False
def zeller(y, m, d):  # ツェラーの公式
    # 土曜日 -> 0
    if m<=2:
        m += 12
        y -= 1
    C, Y = divmod(y, 100)
    h = (d + 26*(m+1)//10 + Y + Y//4 + (-2*C+C//4)) % 7
    return h
def md2d(m, d):  # m 月 d 日は 0-indexed で何日目か？
    # 返り値は [0, 365)
    return sum(days[:m]) + d - 1
def all_md():
    for m, ds in enumerate(days[1:], 1):
        for d in range(1, ds+1):
            yield m, d
def all_ymd(y_start, y_end):
    for y in range(y_start, y_end):
        for m, d in all_md(days=days_leap if is_leap_year(y) else days):
            yield y, m, d


class Factoradic:
    # 階乗進数
    # n (0-indexed) 桁目は n+1 進法、すなわち n 桁目の数字は [0, n] の範囲
    # n 桁目の 1 は n! を表す
    # 検証: https://atcoder.jp/contests/arc047/submissions/7436530
    factorial = [1]
    def __init__(self, a):
        self.value = value = []  # 下の位から入れていく
        if isinstance(a, int):
            n = 1
            while a:
                a, m = divmod(a, n)
                value.append(m)
                n += 1
        elif hasattr(a, "__iter__"):
            self.value = list(a)
        else:
            raise TypeError

    def __int__(self):
        res = 0
        f = 1
        for i, v in enumerate(self.value[1:], 1):
            f *= i
            res += v * f
        return res

    def _set_factorial(self, val):
        factorial = Factoradic.factorial
        n = len(factorial)
        f = factorial[-1]
        while f < val:
            f *= n
            factorial.append(f)
            n += 1

    def to_permutation(self, d):
        # [0, d) の順列のうち、辞書順 self 番目 (0_indexed) のものを返す
        # self >= d! の場合は、self mod d! 番目のものを返す
        # O(d log d)
        value = self.value
        value += [0] * (d-len(value))
        res = []
        n = 1 << d.bit_length()
        bit = [i&-i for i in range(n)]  # BIT を 1 で初期化 (1-indexed)
        for v in value[d-1::-1]:
            i, step = 0, n>>1
            while step:  # BIT 上の二分探索
                if bit[i+step] <= v:
                    i += step
                    v -= bit[i]
                else:
                    bit[i+step] -= 1  # 減算も同時に行う
                step >>= 1
            res.append(i)  # i 要素目までの累積和が v 以下になる最大の i
        return res

    def __isub__(self, other):  # other は Factoradic 型
        value = self.value
        value_ = other.value
        value += [0] * (len(value_)-len(value))
        m = 0  # 繰り下がり
        for i, v in enumerate(value_[1:]+[0]*(len(value)-len(value_)), 1):
            value[i] -= v + m
            if value[i] < 0:
                value[i] += i + 1
                m = 1
            else:
                m = 0
        if m==1:
            assert False
        return self

    def __ifloordiv__(self, other):  # other は int 型
        value = self.value
        m = 0
        for n in range(len(value)-1, -1, -1):
            v = value[n] + m
            value[n], m = divmod(v, other)
            m *= n
        return self

def lagrange_interpolation(X, Y, mod):
    # ラグランジュ補間 O(n^2)
    # n 個の条件から n-1 次多項式を作る 返り値は次数の降順
    # 検証: https://atcoder.jp/contests/abc137/submissions/6845025
    # mod を取らない場合 scipy.interpolate.lagrange が使えそう
    n = len(X)
    g = [0]*(n+1)
    g[0] = 1
    for i, x in enumerate(X):
        for j in range(i, -1, -1):
            g[j+1] += g[j] * (-x) % mod
    res = [0]*n
    for x, y in zip(X, Y):
        f = g[:]
        denom = 0
        v = 1
        pow_x = [1]  # x の idx 乗
        for _ in range(n-1):
            v = v * x % mod
            pow_x.append(v)
        pow_x.reverse()  # n-1 乗 ~ 0 乗
        for i, po in enumerate(pow_x):
            f_i = f[i]
            f[i+1] += f_i * x % mod  # f = g / (x - x_i) を組立除法で求める
            denom = (denom + f_i * po) % mod
        denom_inv = pow(denom, mod-2, mod)
        for i, f_i in enumerate(f[:n]):
            res[i] += (f_i * y * denom_inv)# % mod  # mod が大きいと 64bit に収まらなくなるのでひとつずつ mod 取った方がいいか？
    return [v % mod for v in res]


# karatsuba 法
def list2bigint(lst, bit=64):  # 非負整数のみ
    fmt = "0{}x".format(bit//4)
    return int("".join(format(v, fmt) for v in lst), 16)

def bigint2list(n, bit=64, length=None):  # length を指定しない場合左側の 0 は省略されるので注意
    n_hex = bit//4
    s = format(n, "0{}x".format(n_hex*length) if length else "x")
    s = -len(s) % n_hex * "0" + s
    return [int(s[i:i+n_hex], 16) for i in range(0, len(s), n_hex)]


def gauss_jordan(A):
    # F2 上の Gauss Jordan の掃き出し法
    # 基底を取り出す
    # 引数を破壊的に変更する
    idx = 0
    for i in range(59, -1, -1):
        for j, a in enumerate(A[idx:], idx):
            if a>>i & 1:
                break
        else:
            continue
        A[idx], A[j] = A[j], A[idx]
        for j in range(len(A)):
            if j != idx and A[j]>>i & 1:
                A[j] ^= a
        idx += 1
    assert not any(A[idx:])
    del A[idx:]


class HeavyLightDecomposition:
    # HL 分解 (HLD)
    # 検証1 (lca) [CF] Tree Array: https://codeforces.com/contest/1540/submission/120730365
    # 検証2 (lca) [yosupo] Lowest Common Ancestor: https://judge.yosupo.jp/submission/51656
    def __init__(self, E, root=1):
        # E は双方向に辺を張った木で、破壊的に有向グラフに変更される
        # O(N)
        self.E = E
        self.root = root
        self.N = N = len(E)  # 頂点数
        self.Parent = [-1] * N  # 頂点番号 v -> 親ノード
        self.Size = [-1] * N  # 頂点番号 v -> 部分木のサイズ
        self.dfs1()

        self.Mapping = [-1] * N  # 頂点番号 v -> 内部インデックス
        self.Head = list(range(N))  # 頂点番号 v -> v を含む heavy path の左端の頂点番号
        self.Depth = [0] * N  # 頂点番号 v -> 深さ（root から v までの距離）
        self.dfs2()

    def dfs1(self):
        E = self.E
        Parent, Size = self.Parent, self.Size
        Path = [self.root]
        Idx_edge = [0]
        while Path:
            v = Path[-1]
            idx_edge = Idx_edge[-1]
            Ev = E[v]
            if idx_edge != len(Ev):
                # 行きがけ・通りがけ 辺の数だけ実行される
                u = Ev[idx_edge]
                Idx_edge[-1] += 1
                E[u].remove(v)  # 有向グラフならここをコメントアウトする
                Parent[u] = v
                Path.append(u)
                Idx_edge.append(0)
            else:
                # 帰りがけ 頂点の数だけ実行される
                if len(Ev) >= 2:
                    ma = -1
                    argmax = None
                    for i, u in enumerate(Ev):
                        if Size[u] > ma:
                            ma = Size[u]
                            argmax = i
                    Ev[0], Ev[argmax] = Ev[argmax], Ev[0]
                Size[v] = sum(Size[u] for u in Ev) + 1
                Path.pop()
                Idx_edge.pop()
    
    def dfs2(self):
        E = self.E
        Mapping = self.Mapping
        Head = self.Head
        Depth = self.Depth
        k = 0
        St = [self.root]
        while St:
            v = St.pop()
            Mapping[v] = k
            k += 1
            Ev = E[v]
            if Ev:
                Head[Ev[0]] = Head[v]
                St += Ev[::-1]
                for u in Ev:
                    Depth[u] = Depth[v] + 1  # distance を使わないのならここをコメントアウトする
    
    def lca(self, v, u):
        # O(logN)
        Parent = self.Parent
        Mapping = self.Mapping
        Head = self.Head
        while True:
            if Mapping[v] > Mapping[u]:
                v, u = u, v  # v の方を根に近くする
            if Head[v] == Head[u]:
                return v
            u = Parent[Head[u]]
    
    def distance(self, v, u):
        # O(logN)
        Depth = self.Depth
        return Depth[v] + Depth[u] - 2 * Depth[self.lca(v, u)]


from operator import itemgetter
def dhondt(Votes, n_seats):
    # ドント方式  候補者数 N に対して O(NlogN)
    # 検証1: Logs https://atcoder.jp/contests/abc174/submissions/15637677
    # 検証2: Lake https://atcoder.jp/contests/snuke21/submissions/14785590
    n_candidates = len(Votes)
    sum_votes = sum(Votes)
    S0 = []  # 端数切り捨てによってすぐ確定できる席数
    C = []
    for i, v in enumerate(Votes):
        s0 = n_seats*v//sum_votes
        S0.append(s0)
        for s in range(s0+1, s0+10000):
            if s*sum_votes >= v*(n_seats+n_candidates):
                break
            C.append((s/v, i))
        else:
            assert False
    C.sort(key=itemgetter(0))
    S = S0[:]
    for _, i in C[:n_seats-sum(S0)]:
        S[i] += 1
    return S0, S


def sqrt_case_manager(N):
    # x ∈ [1, N] に対して、N//x の取りうる値と、その値を取るときの x の範囲（半開区間）を列挙
    # O(sqrt(N))
    l = 1
    while l <= N:
        r = N//(N//l) + 1
        yield N//l, l, r
        l = r


def minimum_enclosing_circle(points):
    # 最小包含円 O(N)
    # 返り値は中心の座標と半径
    # 参考: https://tubo28.me/compprog/algorithm/minball/
    # 検証: https://atcoder.jp/contests/abc151/submissions/15834319
    from random import sample
    N = len(points)
    if N == 1:
        return points[0], 0
    points = sample(points, N)
    def cross(a, b):
        return a.real * b.imag - a.imag * b.real
    def norm2(a):
        return a.real * a.real + a.imag * a.imag
    def make_circle_3(a, b, c):
        A, B, C = norm2(b-c), norm2(c-a), norm2(a-b)
        S = cross(b-a, c-a)
        p = (A*(B+C-A)*a + B*(C+A-B)*b + C*(A+B-C)*c) / (4*S*S)
        radius = abs(p-a)
        return p, radius
    def make_circle_2(a, b):
        c = (a+b) / 2
        radius = abs(a-c)
        return c, radius
    def in_circle(point, circle):
        return abs(point-circle[0]) <= circle[1]+1e-7
    p0 = points[0]
    circle = make_circle_2(p0, points[1])
    for i, p_i in enumerate(points[2:], 2):
        if not in_circle(p_i, circle):
            circle = make_circle_2(p0, p_i)
            for j, p_j in enumerate(points[1:i], 1):
                if not in_circle(p_j, circle):
                    circle = make_circle_2(p_i, p_j)
                    for p_k in points[:j]:
                        if not in_circle(p_k, circle):
                            circle = make_circle_3(p_i, p_j, p_k)
    return circle


class Osa_k:
    def __init__(self, n_max):
        self.min_factor = min_factor = list(range(n_max+1))
        for i in range(2, int(n_max**0.5)+1):
            if min_factor[i] == i:
                for j in range(i*i, n_max+1, i):
                    if min_factor[j] == j:
                        min_factor[j] = i

    def __call__(self, n):
        min_factor = self.min_factor
        n_twoes = (n & -n).bit_length() - 1  # 最悪ケースでは速くなる
        res = [2] * n_twoes
        n >>= n_twoes
        while n > 1:
            p = min_factor[n]
            res.append(p)
            n //= p
        return res


def is_odd_permutation(A):
    # [0, N) の順列が奇置換であるかを返す
    # 参考: https://atcoder.jp/contests/chokudai_S001/submissions/5745441
    A_ = A[:]
    res = 0
    for idx in range(len(A)):
        a = A_[idx]
        while a != idx:
            A_[idx], A_[a] = A_[a], A_[idx]
            res += 1
            a = A_[idx]
    return res % 2


def recurrence_relation_1(x, y, a0, n, mod):
    # 二項間漸化式 a_n = x * a_{n-1} + y
    a = a0
    while n:
        n, m = divmod(n, 2)
        if m:
            a = (a * x + y) % mod
        x, y = x * x % mod, (x * y + y) % mod
    return a

def recurrence_relation_2(r, b, c, a0, n, mod):
    # 二項間漸化式 a_n = r * a_{n-1} + b * n + c
    # 検証: https://atcoder.jp/contests/abc129/submissions/16044714
    a = a0
    n_ = 0
    c += b  # これをなくすと a_{n+1} = r * a_n + b * n + c になる
    diff = 1
    while n:
        n, m = divmod(n, 2)
        if m:
            a = (a * r + b * n_ + c) % mod
            n_ += diff
        r, b, c = r*r%mod, (r*b+b)%mod, (r*c+b*diff+c)%mod
        diff <<= 1
    return a


class Polynomial:
    # 多項式
    def __init__(self, coef, mod=10**9+7):
        # 降べきの順
        self.coef = coef
        self.mod = mod

    def __repr__(self):
        return str(self.coef)

    def __add__(self, other):
        from itertools import zip_longest
        coef1, coef2, mod = self.coef, other.coef, self.mod
        res = [(c1 + c2) % mod for c1, c2 in zip_longest(coef1[::-1], coef2[::-1], fillvalue=0)]
        while len(res) > 0 and res[-1] == 0:
            del res[-1]
        res.reverse()
        return Polynomial(res, mod=mod)

    def __mul__(self, other):
        if isinstance(other, int):
            other = Polynomial([other])
        coef1, coef2, mod = self.coef, other.coef, self.mod
        res = [0] * (len(coef1) + len(coef2) - 1)
        for i, c1 in enumerate(coef1):
            for j, c2 in enumerate(coef2, i):
                res[j] = (res[j] + c1 * c2) % mod
        return Polynomial(res, mod=mod)

    def __divmod__(self, other):
        coef1, coef2, mod = self.coef[:], other.coef, self.mod
        assert coef2[0] == 1
        quotient = []
        n = len(coef1) - len(coef2) + 1
        if n < 0:
            return Polynomial([], mod=mod), Polynomial(coef1, mod=mod)
        for i in range(n):
            r = coef1[i]
            quotient.append(r)
            for j, c2 in enumerate(coef2, i):  # enumerate(coef2[1:], i+1) でもいい
                coef1[j] = (coef1[j] - r * c2) % mod
        return Polynomial(quotient, mod=mod), Polynomial(coef1[n:], mod=mod)

    def __imod__(self, other):
        coef1, coef2, mod = self.coef, other.coef, self.mod
        n = len(coef1) - len(coef2) + 1
        if n < 0:
            return self
        for i in range(n):
            r = coef1[i]
            for j, c2 in enumerate(coef2, i):
                coef1[j] = (coef1[j] - r * c2) % mod
        self.coef = coef1[n:]
        return self

def kitamasa(C, n, mod=10**9+7):
    # C: 係数（a_n = C[0] * a_{n-1} + C[1] * a{n-2} + ...）
    # n: 一番小さい初期値が a_i で求めたい項が a_j なら j-k
    Q = Polynomial([1] + [-c for c in C], mod=mod)
    res = Polynomial([1], mod=mod)
    X = Polynomial([1, 0], mod=mod)
    while n:
        n, r = divmod(n, 2)
        if r:
            res = res * X
            res %= Q
        X = X * X
        X %= Q
    return res.coef


class MaxClique:
    # 最大クリーク
    # Bron–Kerbosch algorithm (O(1.4422^|V|)) の枝刈りをしたもの
    # 参考: https://atcoder.jp/contests/code-thanks-festival-2017-open/submissions/2691674
    # 検証1: https://atcoder.jp/contests/code-thanks-festival-2017-open/submissions/7620028
    # 検証2: https://judge.yosupo.jp/submission/12486
    def __init__(self, n):
        self.n = n
        self.E = [0] * n
        self.stk = [0] * n  # 最大クリークが入るスタック

    def __repr__(self):
        return "\n".join("{:0{}b}".format(e, self.n) for e in self.E)

    def add_edge(self, v, u):
        # assert v != u
        self.E[v] |= 1 << u
        self.E[u] |= 1 << v

    def invert(self):
        # 補グラフにする
        n, E = self.n, self.E
        mask = (1<<n) - 1  # 正の数にしないと popcount がバグる
        for i in range(n):
            E[i] = ~E[i] & (mask ^ 1<<i)  # 自己ループがあるとバグる

    def solve(self):
        n, E = self.n, self.E
        deg = [bin(v).count("1") for v in E]
        self.index = index = sorted(range(n), key=lambda x: deg[x], reverse=True)  # 頂点番号を次数の降順にソート
        self.E_sorted = E_sorted = []  # E を 次数の降順に並び替えたもの
        for v in index:
            E_sorted_ = 0
            E_v = E[v]
            for i, u in enumerate(index):
                if E_v >> u & 1:
                    E_sorted_ |= 1 << i
            E_sorted.append(E_sorted_)
        cand = (1 << n) - 1  # 候補の集合
        self.cans = 1  # 最大クリークを構成する集合  # 復元するときはこれを使う
        self.ans = 1
        self._dfs(0, cand)
        return self.ans

    def _dfs(self, elem_num, candi):
        if self.ans < elem_num:
            self.ans = elem_num
            cans_ = 0
            index = self.index
            for s in self.stk[:elem_num]:
                cans_ |= 1 << index[s]
            self.cans = cans_
        potential = elem_num + bin(candi).count("1")
        if potential <= self.ans:
            return
        E_sorted = self.E_sorted
        pivot = (candi & -candi).bit_length() - 1  # 候補から頂点をひとつ取り出す
        smaller_candi = candi & ~E_sorted[pivot]  # pivot と直接結ばれていない頂点の集合（自己ループの無いグラフなので pivot を含む）
        while smaller_candi and potential > self.ans:
            next = smaller_candi & -smaller_candi
            candi ^= next
            smaller_candi ^= next
            potential -= 1
            next = next.bit_length() - 1
            if next == pivot or smaller_candi & E_sorted[next]:
                self.stk[elem_num] = next
                self._dfs(elem_num + 1, candi & E_sorted[next])


def rerooting(n, edges, identity, merge, add_node):
    # 全方位木 dp
    # 参考1: https://qiita.com/keymoon/items/2a52f1b0fb7ef67fb89e
    # 参考2: https://atcoder.jp/contests/abc160/submissions/15255726
    # 検証: Distributing Integers https://atcoder.jp/contests/abc160/submissions/15971070
    from functools import reduce
    G = [[] for _ in range(n)]
    for a, b in edges:
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
    dp_down = [0] * n  # 自身とその下
    for v in order[:0:-1]:
        dp_down[v] = add_node(reduce(
            merge, (dp_down[u] for u in G[v]), identity
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
    results = [add_node(
        reduce(merge, (dp_down[u] for u in Gv), dp_up[v]), v
    ) for v, Gv in enumerate(G)]
    return results


class SlidingMinimum:
    # スライド最小値
    # 検証: https://atcoder.jp/contests/typical90/submissions/22738794
    def __init__(self):
        from collections import deque
        self.left = self.right = 0
        self.q = deque()

    def push(self, val):
        q = self.q
        while q and q[-1][0] >= val:
            q.pop()
        q.append((val, self.right))
        self.right += 1

    def pop(self):
        if self.q[0][1] == self.left:
            self.q.popleft()
        self.left += 1

    def min(self):
        return self.q[0][0] if self.q else 1<<62

    def __len__(self):
        return self.right - self.left

class Doubling:
    # ダブリング
    def __init__(self, nexts, max_n):
        self.table = [nexts[:]]
        for k in range(max_n.bit_length()-1):
            perm = self.table[-1]
            perm_next = [0] * (len(nexts))
            for idx_perm, p in enumerate(perm):
                perm_next[idx_perm] = perm[p]  # perm[p] == perm[perm[idx_perm]]
            self.table.append(perm_next)

    def get(self, idx, n):
        for bit, t in enumerate(self.table):
            if n >> bit & 1:
                idx = t[idx]
        return idx

class DoublingAggregation:
    # ダブリング + 集約
    # 検証1: Sequence Sum https://atcoder.jp/contests/abc179/submissions/17083107
    # 検証2: Keep Distances https://atcoder.jp/contests/acl1/submissions/17084246
    def __init__(self, nexts, arr, max_n, op, e):
        # op はモノイド
        n = len(nexts)
        self.table = [nexts[:]]
        self.data = [arr[:]]
        self.op = op
        self.e = e
        self.max_n = max_n
        for k in range(max_n.bit_length()-1):
            perm = self.table[-1]
            perm_next = []
            dat = self.data[-1]
            dat_next = []
            for p, d in zip(perm, dat):
                perm_next.append(perm[p])  # perm[p] == perm[perm[idx_perm]]
                dat_next.append(op(d, dat[p]))
            self.table.append(perm_next)
            self.data.append(dat_next)
 
    def prod(self, idx, n):
        # arr[idx] * arr[nexts[idx]] * arr[nexts[nexts[idx]] * ... を n 回繰り替えした値を返す
        val = self.e
        op = self.op
        for bit, (t, dat) in enumerate(zip(self.table, self.data)):
            if n >> bit & 1:
                val = op(val, dat[idx])
                idx = t[idx]
        return idx, val
 
    def max_right(self, idx, f):
        # f(arr[idx] * arr[nexts[idx]] * arr[nexts[nexts[idx]] * ... (n 回)) が
        # True である最大の n と、そのときの prod(idx, n)
        n = 0
        val = self.e
        op = self.op
        for bit, t, dat in zip(range(len(self.table)-1, -1, -1), self.table[::-1], self.data[::-1]):
            val_next = op(val, dat[idx])
            if f(val_next):
                val = val_next
                idx = t[idx]
                n |= 1 << bit
        if n > self.max_n:
            n = self.max_n
        return n, idx, val


def berlekamp_massey(arr, mod):
    # 返り値は数列の後ろに掛けるやつが前（伝われ）
    # 参考: https://judge.yosupo.jp/submission/33639
    n = len(arr)
    bs, cs = [1], [1]
    num_zeros_before_top_of_bs = 0
    y = 1
    l = 0
    for ed in range(n):
        num_zeros_before_top_of_bs += 1
        len_cs, len_bs = len(cs), len(bs) + num_zeros_before_top_of_bs
        x = 0
        for i, c in enumerate(cs):
            x = (x + c * arr[ed-i]) % mod
        if x == 0:
            continue
        freq = x * pow(y, mod-2, mod) % mod
        if len_cs < len_bs:
            if 2 * l <= ed:
                cs_old = cs[:]
                cs += [0] * (len_bs - len_cs)
                for i, b in enumerate(bs, num_zeros_before_top_of_bs):
                    cs[i] = (cs[i] - b * freq) % mod
                bs = cs_old
                l = ed + 1 - l
                num_zeros_before_top_of_bs = 0
                y = x
                continue
            cs += [0] * (len_bs - len_cs)
        for i, b in enumerate(bs, num_zeros_before_top_of_bs):
            cs[i] = (cs[i] - freq * b) % mod
    return [-c % mod for c in cs[1:]]


def karatsuba(poly1, poly2):
    # 多項式乗算 (そんなに速くない)
    n = len(poly1) + len(poly2) - 1
    r = max(len(poly1), len(poly2))
    r = 1 << (r-1).bit_length()  # r 以上の最小の 2 冪
    poly1 = poly1 + [0] * (r - len(poly1))
    poly2 = poly2 + [0] * (r - len(poly2))
    res = [0] * (r * 4)
    path = [[0, r, 0]]
    PHASE = 2
    while path:
        l, r, phase = path[-1]
        c = l + r >> 1
        d = r - l
        half = d >> 1
        if phase == 0:
            if d <= 32:
                for i in range(d):
                    for j in range(d):
                        res[d+d+i+j] += poly1[l+i] * poly2[l+j]
                path.pop()
                continue
            path[-1][PHASE] = 1
            path.append([l, c, 0])
        elif phase == 1:
            for i in range(d, d*2-1):
                res[d+i] += res[i]
                res[d+half+i] += res[i]
                res[i] = 0
            path[-1][PHASE] = 2
            path.append([c, r, 0])
        elif phase == 2:
            for i in range(d, d*2-1):
                res[d+half+i] += res[i]
                res[d+d+i] += res[i]
                res[i] = 0
            for i in range(l, c):
                poly1[half+i] -= poly1[i]
                poly2[half+i] -= poly2[i]
            path[-1][PHASE] = 3
            path.append([c, r, 0])
        else:
            for i in range(l, c):
                poly1[half+i] += poly1[i]
                poly2[half+i] += poly2[i]
            for i in range(d, d*2-1):
                res[d+half+i] -= res[i]
                res[i] = 0
            path.pop()
    return res[len(res)//2:len(res)//2+n]


# リスト埋め込み用  # AtCoder なら 50000 要素くらいは埋め込める  # 圧縮率が高ければそれ以上も埋め込める
def encode_list(lst):
    import array, gzip, base64
    int32 = "l" if array.array("l").itemsize == 4 else "i"
    return base64.b64encode(gzip.compress(array.array(int32, lst)))

def decode_list(lst):
    import array, gzip, base64
    int32 = "l" if array.array("l").itemsize == 4 else "i"
    return array.array(int32, gzip.decompress(base64.b64decode(lst)))



