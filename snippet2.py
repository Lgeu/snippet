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


# リスト埋め込み用  # AtCoder なら 50000 要素くらいは埋め込める  # 圧縮率が高ければそれ以上も埋め込める
def encode_list(lst):
    import array, gzip, base64
    int32 = "l" if array.array("l").itemsize == 4 else "i"
    return base64.b64encode(gzip.compress(array.array(int32, lst)))

def decode_list(lst):
    import array, gzip, base64
    int32 = "l" if array.array("l").itemsize == 4 else "i"
    return array.array(int32, gzip.decompress(base64.b64decode(lst)))



