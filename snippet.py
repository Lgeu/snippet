# 拡張ユークリッド互除法
# ax + by = gcd(a,b)の最小整数解を返す
# 最大公約数はg
def egcd(a, b):
    if a == 0:
        return b, 0, 1
    else:
        g, y, x = egcd(b % a, a)
        return g, x - (b // a) * y, y

def modinv(a, mod=10**9+7):
    return pow(a, mod-2, mod)

"""
# mを法とするaの乗法的逆元
def modinv(a, m):
    g, x, y = egcd(a, m)
    if g != 1:
        raise Exception('modular inverse does not exist')
    else:
        return x % m
"""

# nCr mod m
# modinvが必要
# rがn/2に近いと非常に重くなる
def combination(n, r, mod=10**9+7):
    r = min(r, n-r)
    res = 1
    for i in range(r):
        res = res * (n - i) * modinv(i+1, mod) % mod
    return res

# nHr mod m
def H(n, r, mod=10**9+7):
    return combination(n+r-1, r, mod)

# nCrをすべてのr(0<=r<=n)について求める
# nC0, nC1, nC2, ... , nCn を求める
# modinvが必要
def combination_list(n, mod=10**9+7):
    lst = [1]
    for i in range(1, n+1):
        lst.append(lst[-1] * (n+1-i) % mod * modinv(i, mod) % mod)
    return lst

# 階乗のmod逆元のリストを返す O(n)
def facinv_list(n, mod=10**9+7):
    L = [1]
    for i in range(1, n+1):
        L.append(L[i-1] * modinv(i, mod) % mod)
    return L


class Combination:
    """
    O(n)の前計算を1回行うことで，O(1)でnCr mod mを求められる
    n_max = 10**6のとき前処理は約950ms (PyPyなら約340ms, 10**7で約1800ms)
    使用例：
    comb = Combination(1000000)
    print(comb(5, 3))  # 10
    """
    def __init__(self, n_max, mod=10**9+7):
        self.mod = mod
        self.modinv = self.make_modinv_list(n_max)
        self.fac, self.facinv = self.make_factorial_list(n_max)

    def __call__(self, n, r):
        return self.fac[n] * self.facinv[r] % self.mod * self.facinv[n-r] % self.mod

    def make_factorial_list(self, n):
        # 階乗のリストと階乗のmod逆元のリストを返す O(n)
        # self.make_modinv_list()が先に実行されている必要がある
        fac = [1]
        facinv = [1]
        for i in range(1, n+1):
            fac.append(fac[i-1] * i % self.mod)
            facinv.append(facinv[i-1] * self.modinv[i] % self.mod)
        return fac, facinv

    def make_modinv_list(self, n):
        # 0からnまでのmod逆元のリストを返す O(n)
        modinv = [0] * (n+1)
        modinv[1] = 1
        for i in range(2, n+1):
            modinv[i] = self.mod - self.mod//i * modinv[self.mod%i] % self.mod
        return modinv


# nまでの自然数が素数かどうかを表すリストを返す
def makePrimeChecker(n):
    isPrime = [True] * (n + 1)
    isPrime[0] = False
    isPrime[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if not isPrime[i]:
            continue
        for j in range(i * 2, n + 1, i):
            isPrime[j] = False
    return isPrime

# 素因数分解
def prime_decomposition(n):
    i = 2
    table = []
    while i * i <= n:
        while n % i == 0:
            n //= i
            table.append(i)
        i += 1
    if n > 1:
        table.append(n)
    return table


def full(L):  # 並び替えをすべて挙げる，全探索用
    # これitertools.permutationsでええやん！！！！！！！
    if len(L) == 1:
        return [L]
    else:
        L2 = []
        for i in range(len(L)):
            L2.extend([[L[i]] + Lc for Lc in full(L[:i] + L[i+1:])])
        return L2
"""
from itertools import permutations
"""

# 転倒数
def mergecount(A):
    cnt = 0
    n = len(A)
    if n>1:
        A1 = A[:n>>1]
        A2 = A[n>>1:]
        cnt += mergecount(A1)
        cnt += mergecount(A2)
        i1=0
        i2=0
        for i in range(n):
            if i2 == len(A2):
                A[i] = A1[i1]
                i1 += 1
            elif i1 == len(A1):
                A[i] = A2[i2]
                i2 += 1
            elif A1[i1] <= A2[i2]:
                A[i] = A1[i1]
                i1 += 1
            else:
                A[i] = A2[i2]
                i2 += 1
                cnt += n//2 - i1
    return cnt

# Binary Indexed Tree
class Bit:
    """
    0-indexed
    # 使用例
    bit = Bit(10)  # 要素数
    bit.add(2, 10)
    print(bit.sum(5))  # 10
    """
    def __init__(self, n):
        self.size = n
        self.tree = [0]*(n+1)

    def __iter__(self):
        psum = 0
        for i in range(self.size):
            csum = self.sum(i+1)
            yield csum - psum
            psum = csum
        raise StopIteration()

    def __str__(self):  # O(nlogn)
        return str(list(self))

    def sum(self, i):
        # [0, i) の要素の総和を返す
        if not (0 <= i <= self.size): raise ValueError("error!")
        s = 0
        while i>0:
            s += self.tree[i]
            i -= i & -i
        return s

    def add(self, i, x):
        if not (0 <= i < self.size): raise ValueError("error!")
        i += 1
        while i <= self.size:
            self.tree[i] += x
            i += i & -i

    def __getitem__(self, key):
        if not (0 <= key < self.size): raise IndexError("error!")
        return self.sum(key+1) - self.sum(key)

    def __setitem__(self, key, value):
        # 足し算と引き算にはaddを使うべき
        if not (0 <= key < self.size): raise IndexError("error!")
        self.add(key, value - self[key])

class BitImos:
    """
    ・範囲すべての要素に加算
    ・ひとつの値を取得
    の2種類のクエリをO(logn)で処理
    """
    def __init__(self, n):
        self.bit = Bit(n+1)

    def add(self, s, t, x):
        # [s, t)にxを加算
        self.bit.add(s, x)
        self.bit.add(t, -x)

    def get(self, i):
        return self[i]

    def __getitem__(self, key):
        # 位置iの値を取得
        return self.bit.sum(key+1)

"""
" BITで転倒数を求められる
A = [3, 10, 1, 8, 5, 5, 1]
bit = Bit(max(A)+1)
ans = 0
for i, a in enumerate(A):
    ans += i - bit.sum(a+1)
    bit.add(a, 1)
print(ans)
"""
import heapq
class Dijkstra:
    # 計算量 O((E+V)logV)

    # adjは2次元defaultdict
    def dijkstra(self, adj, start, goal=None):

        num = len(adj)  # グラフのノード数
        dist = [float('inf') for i in range(num)]  # 始点から各頂点までの最短距離を格納する
        prev = [float('inf') for i in range(num)]  # 最短経路における，その頂点の前の頂点のIDを格納する

        dist[start] = 0
        q = []  # プライオリティキュー．各要素は，(startからある頂点vまでの仮の距離, 頂点vのID)からなるタプル
        heapq.heappush(q, (0, start))  # 始点をpush

        while len(q) != 0:
            prov_cost, src = heapq.heappop(q)  # pop

            # プライオリティキューに格納されている最短距離が，現在計算できている最短距離より大きければ，distの更新をする必要はない
            if dist[src] < prov_cost:
                continue

            # 他の頂点の探索
            for dest, cost in adj[src].items():
                if dist[dest] > dist[src] + cost:
                    dist[dest] = dist[src] + cost  # distの更新
                    heapq.heappush(q, (dist[dest], dest))  # キューに新たな仮の距離の情報をpush
                    prev[dest] = src  # 前の頂点を記録

        if goal is not None:
            return self.get_path(goal, prev)
        else:
            return dist

    def get_path(self, goal, prev):
        path = [goal]  # 最短経路
        dest = goal

        # 終点から最短経路を逆順に辿る
        while prev[dest] != float('inf'):
            path.append(prev[dest])
            dest = prev[dest]

        # 経路をreverseして出力
        return list(reversed(path))
#from collections import defaultdict
#E = defaultdict(lambda: defaultdict(lambda: float("inf")))


# UF木
class Uf:
    def __init__(self):
        self.Par = list(range(N + 1))

    def root(self, x):
        if self.Par[x] == x:
            return x
        else:
            self.Par[x] = self.root(self.Par[x])
            return self.Par[x]

    def same(self, x, y):
        return self.root(x) == self.root(y)

    def unite(self, x, y):
        x = self.root(x)
        y = self.root(y)
        if x != y:
            self.Par[x] = y

"""
# UF木
Par = list(range(N))
def root(x):
    if Par[x] == x:
        return x
    else:
        Par[x] = root(Par[x])
        return Par[x]

def same(x, y):
    return root(x) == root(y)

def unite(x, y):
    x = root(x)
    y = root(y)
    if x != y:
        Par[x] = y


# ----
"""

def norm(x1, y1, x2, y2):
    return ((x1-x2)**2 + (y1-y2)**2)**0.5
def d(a, b, c, x, y):
    return abs(a*x + b*y + c) / (a**2 + b**2)**0.5
def points2line(x1, y1, x2, y2):
    la = y1 - y2
    lb = x2 - x1
    lc = x1 * (y2 - y1) + y1 * (x1 - x2)
    return la, lb, lc


"""

import sys
def input():  # 入力10**6行あたり約70ms早くなる inputの遅いPyPyでは約180ms
    return sys.stdin.readline()[:-1]

from functools import lru_cache
@lru_cache(maxsize=None)  # メモ化再帰したい関数の前につける

import sys
sys.setrecursionlimit(500000)
#from operator import itemgetter
from collections import defaultdict
from itertools import product  # 直積

ord("a") - 97  # chr

N = int(input())
N, K = map(int, input().split())
L = [int(input()) for i in range(N)]
A = list(map(int, input().split()))
S = [list(map(int, input().split())) for i in range(H)]
"""
