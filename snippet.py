def egcd(a, b):
    # 拡張ユークリッド互除法
    # ax + by = gcd(a,b)の最小整数解を返す
    # 最大公約数はg
    if a == 0:
        return b, 0, 1
    else:
        g, y, x = egcd(b % a, a)
        return g, x - (b // a) * y, y

def chineseRem(b1, m1, b2, m2):
    # 中国剰余定理
    # x ≡ b1 (mod m1) ∧ x ≡ b2 (mod m2) <=> x ≡ r (mod m)
    # となる(r. m)を返す
    # 解無しのとき(0, -1)
    d, p, q = egcd(m1, m2)
    if (b2 - b1) % d != 0:
        return 0, -1
    m = m1 * (m2 // d)  # m = lcm(m1, m2)
    tmp = (b2-b1) // d * p % (m2 // d)
    r = (b1 + m1 * tmp) % m
    return r, m

def modinv(a, mod=10**9+7):
    return pow(a, mod-2, mod)

def combination(n, r, mod=10**9+7):
    # nCr mod m
    # rがn/2に近いと非常に重くなる
    n1, r = n+1, min(r, n-r)
    numer = denom = 1
    for i in range(1, r+1):
        numer = numer * (n1-i) % mod
        denom = denom * i % mod
    return numer * pow(denom, mod-2, mod) % mod

def H(n, r, mod=10**9+7):
    # nHr mod m
    return combination(n+r-1, r, mod)

def combination_list(n, mod=10**9+7):
    # nCrをすべてのr(0<=r<=n)について求める
    # nC0, nC1, nC2, ... , nCn を求める
    lst = [1]
    for i in range(1, n+1):
        lst.append(lst[-1] * (n+1-i) % mod * pow(i, mod-2, mod) % mod)
    return lst

def make_modinv_list(n, mod=10**9+7):
    # 0 から n までの mod 逆元のリストを返す O(n)
    modinv = [0, 1]
    for i in range(2, n+1):
        modinv.append(mod - mod//i * modinv[mod%i] % mod)
    return modinv

class Combination:
    def __init__(self, n_max, mod=10**9+7):
        # O(n_max + log(mod))
        self.mod = mod
        f = 1
        self.fac = fac = [f]
        for i in range(1, n_max+1):
            f = f * i % mod
            fac.append(f)
        f = pow(f, mod-2, mod)
        self.facinv = facinv = [f]
        for i in range(n_max, 0, -1):
            f = f * i % mod
            facinv.append(f)
        facinv.reverse()

    def __call__(self, n, r):  # self.C と同じ
        return self.fac[n] * self.facinv[r] % self.mod * self.facinv[n-r] % self.mod

    def C(self, n, r):
        if not 0 <= r <= n: return 0
        return self.fac[n] * self.facinv[r] % self.mod * self.facinv[n-r] % self.mod

    def P(self, n, r):
        if not 0 <= r <= n: return 0
        return self.fac[n] * self.facinv[n-r] % self.mod

    def H(self, n, r):
        if (n == 0 and r > 0) or r < 0: return 0
        return self.fac[n+r-1] * self.facinv[r] % self.mod * self.facinv[n-1] % self.mod

    # "n 要素" は区別できる n 要素
    # "k グループ" はちょうど k グループ

    def rising_factorial(self, n, r):  # 上昇階乗冪 n * (n+1) * ... * (n+r-1)
        return self.fac[n+r-1] * self.facinv[n-1] % self.mod

    def stirling_first(self, n, k):  # 第 1 種スターリング数  lru_cache を使うと O(nk)  # n 要素を k 個の巡回列に分割する場合の数
        if n == k: return 1
        if k == 0: return 0
        return (self.stirling_first(n-1, k-1) + (n-1)*self.stirling_first(n-1, k)) % self.mod

    def stirling_second(self, n, k):  # 第 2 種スターリング数 O(k + log(n))  # n 要素を区別のない k グループに分割する場合の数
        if n == k: return 1  # n==k==0 のときのため
        return self.facinv[k] * sum((-1)**(k-m) * self.C(k, m) * pow(m, n, self.mod) for m in range(1, k+1)) % self.mod

    def balls_and_boxes_3(self, n, k):  # n 要素を区別のある k グループに分割する場合の数  O(k + log(n))
        return sum((-1)**(k-m) * self.C(k, m) * pow(m, n, self.mod) for m in range(1, k+1)) % self.mod

    def bernoulli(self, n):  # ベルヌーイ数  lru_cache を使うと O(n**2 * log(mod))
        if n == 0: return 1
        if n % 2 and n >= 3: return 0  # 高速化
        return (- pow(n+1, self.mod-2, self.mod) * sum(self.C(n+1, k) * self.bernoulli(k) % self.mod for k in range(n))) % self.mod

    def faulhaber(self, k, n):  # べき乗和 0^k + 1^k + ... + (n-1)^k
        # bernoulli に lru_cache を使うと O(k**2 * log(mod))  bernoulli が計算済みなら O(k * log(mod))
        return pow(k+1, self.mod-2, self.mod) * sum(self.C(k+1, j) * self.bernoulli(j) % self.mod * pow(n, k-j+1, self.mod) % self.mod for j in range(k+1)) % self.mod

    def lah(self, n, k):  # n 要素を k 個の空でない順序付き集合に分割する場合の数  O(1)
        return self.C(n-1, k-1) * self.fac[n] % self.mod * self.facinv[k] % self.mod

    def bell(self, n, k):  # n 要素を k グループ以下に分割する場合の数  O(k**2 + k*log(mod))
        return sum(self.stirling_second(n, j) for j in range(1, k+1)) % self.mod

def make_prime_checker(n):
    # n までの自然数が素数かどうかを表すリストを返す  O(nloglogn)
    is_prime = [False, True, False, False, False, True] * (n//6+1)
    del is_prime[n+1:]
    is_prime[1:4] = False, True, True
    for i in range(5, int(n**0.5)+1):
        if is_prime[i]:
            is_prime[i*i::i] = [False] * (n//i-i+1)
    return is_prime

def prime_factorization(n):
    # 素因数分解
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

def fast_prime_factorization(n):
    # 素因数分解（ロー法）  O(n^(1/4) polylog(n))
    from subprocess import Popen, PIPE
    return list(map(int, Popen(["factor", str(n)], stdout=PIPE).communicate()[0].split()[1:]))

def fast_prime_factorization_many(lst):
    # 素因数分解（ロー法、複数）
    from subprocess import Popen, PIPE
    res = Popen(["factor"] + list(map(str, lst)), stdout=PIPE).communicate()[0].split(b"\n")[:-1]
    return [list(map(int, r.split()[1:])) for r in res]

def miller_rabin(n):
    # 確率的素数判定（ミラーラビン素数判定法）
    # 素数なら確実に True を返す、合成数なら確率的に False を返す
    # True が返ったなら恐らく素数で、False が返ったなら確実に合成数である
    # 参考: http://tjkendev.github.io/procon-library/python/prime/probabilistic.html
    # 検証: https://yukicoder.me/submissions/381948
    primes = [2, 325, 9375, 28178, 450775, 9780504, 1795265022]  # 32bit: [2, 7, 61]
    if n==2: return True
    if n<=1 or n&1==0: return False
    d = m1 = n-1
    d //= d & -d
    for a in primes:
        if a >= n: return True
        t, y = d, pow(a, d, n)
        while t!=m1 and y!=1 and y!=m1:
            y = y * y % n
            t <<= 1
        if y!=m1 and t&1==0: return False
    return True


class Bit:
    # Binary Indexed Tree
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
# BITで転倒数を求められる
A = [3, 10, 1, 8, 5, 5, 1]
bit = Bit(max(A)+1)
ans = 0
for i, a in enumerate(A):
    ans += i - bit.sum(a+1)
    bit.add(a, 1)
print(ans)
"""

# 未検証
class Bit2:
    def __init__(self, n):
        self.bit0 = Bit(n)
        self.bit1 = Bit(n)

    def add(self, l, r, x):
        # [l, r) に x を足す
        self.bit0.add(l, -x * (l-1))
        self.bit1.add(l, x)
        self.bit0.add(r, x * (r-1))
        self.bit1.add(r, -x)

    def sum(self, l, r):
        res = 0
        res += self.bit0.sum(r) + self.bit1.sum(r) * (r-1)
        res -= self.bit0.sum(l) + self.bit1.sum(l) * (l-1)
        return res


def dijkstra(E, start):
    # ダイクストラ法
    from heapq import heappush, heappop
    N = len(E)
    inf = float("inf")
    dist = [inf] * N
    dist[start] = 0
    q = [(0, start)]
    while q:
        dist_v, v = heappop(q)
        if dist[v] != dist_v:
            continue
        for u, dist_vu in E[v]:
            dist_u = dist_v + dist_vu
            if dist_u < dist[u]:
                dist[u] = dist_u
                heappush(q, (dist_u, u))

def shortest_path_faster_algorithm(E, start):
    # ベルマンフォードの更新があるところだけ更新する感じのやつ
    # O(VE) だが実用上高速
    # E は隣接リスト
    # 検証: http://judge.u-aizu.ac.jp/onlinejudge/review.jsp?rid=4005805#1
    # deque 版 (コーナーケースに強い？): http://judge.u-aizu.ac.jp/onlinejudge/review.jsp?rid=4005813
    inf = float("inf")  # 10**18 は良くない
    N = len(E)
    q = [start]
    distance = [inf] * N;  distance[start] = 0
    in_q = [0] * N;        in_q[start] = True
    times = [0] * N;       times[start] = 1
    while q:
        v = q.pop()
        in_q[v] = False
        dist_v = distance[v]
        for u, cost in E[v]:
            new_dist_u = dist_v + cost
            if distance[u] > new_dist_u:
                times[u] += 1
                if times[u] >= N:  # 負閉路検出
                    distance[u] = -inf
                else:
                    distance[u] = new_dist_u
                if not in_q[u]:
                    in_q[u] = True
                    q.append(u)
    return distance


class UnionFind:
    def __init__(self, N):
        self.p = list(range(N))
        self.rank = [0] * N
        self.size = [1] * N

    def root(self, x):
        if self.p[x] != x:
            self.p[x] = self.root(self.p[x])
        return self.p[x]

    def same(self, x, y):
        return self.root(x) == self.root(y)

    def unite(self, x, y):
        u = self.root(x)
        v = self.root(y)
        if u == v:
            return
        if self.rank[u] < self.rank[v]:
            self.p[u] = v
            self.size[v] += self.size[u]
            self.size[u] = 0
        else:
            self.p[v] = u
            self.size[u] += self.size[v]
            self.size[v] = 0
            if self.rank[u] == self.rank[v]:
                self.rank[u] += 1

    def count(self, x):
        return self.size[self.root(x)]


def fast_zeta_transform_superset(arr):
    # 上位集合の高速ゼータ変換  O(nlog(n))
    # fast_zeta_transform_superset([1]*8) => [8, 4, 4, 2, 4, 2, 2, 1]
    # 添字 and での畳み込みに使う
    n = len(arr)
    assert n & -n == n  # n は 2 冪
    for i in range(n.bit_length()-1):
        for s in range(n):
            if s>>i&1 == 0:
                arr[s] -= arr[s|1<<i]  # -= にすると逆変換
    return arr

def fast_zeta_transform_subset(arr):
    # 下位集合の高速ゼータ変換  O(nlog(n))
    # fast_zeta_transform_subset([1]*8) => [1, 2, 2, 4, 2, 4, 4, 8]
    # 添字 or での畳み込みに使う
    n = len(arr)
    assert n & -n == n  # n は 2 冪
    for i in range(n.bit_length()-1):
        for s in range(n):
            if s>>i&1:
                arr[s] += arr[s^1<<i]  # -= にすると逆変換
    return arr

# 倍数集合の高速ゼータ変換: https://atcoder.jp/contests/agc038/submissions/7671865


class SegmentTree(object):
    # 検証: https://atcoder.jp/contests/nikkei2019-2-qual/submissions/8434117
    # 参考: https://atcoder.jp/contests/abc014/submissions/3935971
    __slots__ = ["elem_size", "tree", "default", "op", "real_size"]

    def __init__(self, a, default, op):
        self.default = default
        self.op = op
        if hasattr(a, "__iter__"):
            self.real_size = len(a)
            self.elem_size = elem_size = 1 << (self.real_size-1).bit_length()
            self.tree = tree = [default] * (elem_size * 2)
            tree[elem_size:elem_size + self.real_size] = a
            for i in range(elem_size - 1, 0, -1):
                tree[i] = op(tree[i << 1], tree[(i << 1) + 1])
        elif isinstance(a, int):
            self.real_size = a
            self.elem_size = elem_size = 1 << (self.real_size-1).bit_length()
            self.tree = [default] * (elem_size * 2)

    def get_value(self, x: int, y: int) -> int:  # 半開区間
        l, r = x + self.elem_size, y + self.elem_size
        tree, result, op = self.tree, self.default, self.op
        while l < r:
            if l & 1:
                result = op(tree[l], result)
                l += 1
            if r & 1:
                r -= 1
                result = op(tree[r], result)
            l, r = l >> 1, r >> 1
        return result

    def set_value(self, i: int, value: int) -> None:
        k = self.elem_size + i
        op, tree = self.op, self.tree
        tree[k] = value
        while k > 1:
            k >>= 1
            tree[k] = op(tree[k << 1], tree[(k << 1) + 1])

    def get_one_value(self, i):
        return self.tree[i+self.elem_size]

    def debug(self):
        print(self.tree[self.elem_size:self.elem_size+self.real_size])


def manacher(S):
    # 最長回文 O(n)
    # R[i] := i 文字目を中心とする最長の回文の半径（自身を含む）
    # 偶数長の回文を検出するには "$a$b$a$a$b$" のようにダミーを挟む
    # 検証: https://atcoder.jp/contests/wupc2019/submissions/8665857
    # 左右で違う条件: https://atcoder.jp/contests/code-thanks-festival-2014-a-open/submissions/12911822
    c, r, n = 0, 0, len(S)  # center, radius, length
    R = [0]*n
    while c < n:
        while c-r >= 0 and c+r < n and S[c-r] == S[c+r]:
            r += 1
        R[c] = r
        d = 1  # distance from center
        while c-d >= 0 and c+d < n and d+R[c-d] < r:
            R[c+d] = R[c-d]
            d += 1
        c += d
        r -= d
    return R

def z_algorithm(S):
    # Z アルゴリズム  O(n)
    # Z[i] := S と S[i:] で prefix が何文字一致しているか
    # 検証: https://atcoder.jp/contests/arc055/submissions/14179788
    i, j, n = 1, 0, len(S)
    Z = [0] * n
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


# 最大流問題
from collections import deque
INF = float("inf")
TO = 0;  CAP = 1;  REV = 2
class Dinic:
    def __init__(self, N):
        self.N = N
        self.V = [[] for _ in range(N)]  # to, cap, rev
        # 辺 e = V[n][m] の逆辺は V[e[TO]][e[REV]]
        self.level = [0] * N

    def add_edge(self, u, v, cap):
        self.V[u].append([v, cap, len(self.V[v])])
        self.V[v].append([u, 0, len(self.V[u])-1])

    def add_edge_undirected(self, u, v, cap):  # 未検証
        self.V[u].append([v, cap, len(self.V[v])])
        self.V[v].append([u, cap, len(self.V[u])-1])

    def bfs(self, s: int) -> bool:
        self.level = [-1] * self.N
        self.level[s] = 0
        q = deque()
        q.append(s)
        while len(q) != 0:
            v = q.popleft()
            for e in self.V[v]:
                if e[CAP] > 0 and self.level[e[TO]] == -1:  # capが1以上で未探索の辺
                    self.level[e[TO]] = self.level[v] + 1
                    q.append(e[TO])
        return True if self.level[self.g] != -1 else False  # 到達可能

    def dfs(self, v: int, f) -> int:
        if v == self.g:
            return f
        for i in range(self.ite[v], len(self.V[v])):
            self.ite[v] = i
            e = self.V[v][i]
            if e[CAP] > 0 and self.level[v] < self.level[e[TO]]:
                d = self.dfs(e[TO], min(f, e[CAP]))
                if d > 0:  # 増加路
                    e[CAP] -= d  # cap を減らす
                    self.V[e[TO]][e[REV]][CAP] += d  # 反対方向の cap を増やす
                    return d
        return 0

    def solve(self, s, g):
        self.g = g
        flow = 0
        while self.bfs(s):  # 到達可能な間
            self.ite = [0] * self.N
            f = self.dfs(s, INF)
            while f > 0:
                flow += f
                f = self.dfs(s, INF)
        return flow


def lis(A: list):  # 最長増加部分列
    # original author: ikatakos
    # https://ikatakos.com/pot/programming_algorithm/dynamic_programming/longest_common_subsequence
    from bisect import bisect_left
    L = [A[0]]
    for a in A[1:]:
        if a > L[-1]:
            # Lの末尾よりaが大きければ増加部分列を延長できる
            L.append(a)
        else:
            # そうでなければ、「aより小さい最大要素の次」をaにする
            # 該当位置は、二分探索で特定できる
            L[bisect_left(L, a)] = a
    return len(L)


class NewtonInterpolation:
    # ニュートン補間  O(n^2)  n は次元
    # 具体的な係数は保持しない
    def __init__(self, X=(), Y=(), mod=10**9+7):
        self.mod = mod
        self.X = []
        self.C = []
        for x, y in zip(X, Y):
            self.add_constraint(x, y)

    def add_constraint(self, x, y):  # O(n)
        mod, X, C = self.mod, self.X, self.C
        numer, denom = y, 1
        for c, x_ in zip(C, X):
            numer -= denom * c
            denom = denom * (x - x_) % mod
        X.append(x)
        C.append(numer * pow(denom, mod-2, mod) % mod)

    def calc(self, x):
        mod, X, C = self.mod, self.X, self.C
        y = 0
        for c, x_ in zip(C[::-1], X[::-1]):
            y = (y * (x - x_) + c) % mod
        return y

def fast_lagrange_interpolation(Y, x, mod=10**9+7):
    # X = [0, 1, 2, ... , n] のラグランジュ補間  O(nlog(mod))  # n==len(Y)-1
    if 0 <= x < len(Y):
        return Y[x] % mod
    factorial, f, numer = [1], 1, x
    for x_ in range(1, len(Y)):
        f = f * x_ % mod
        factorial.append(f)
        numer = numer * (x - x_) % mod
    y = 0
    for x_, (y_, denom1, denom2) in enumerate(zip(Y, factorial, factorial[::-1])):
        y = (y_ * numer * pow((x-x_)*denom1*denom2, mod-2, mod) - y) % mod
    return y

def faulhaber(k, n, mod=10**9+7):  # べき乗和 0^k + 1^k + ... + (n-1)^k
    # n に関する k+1 次式になるので最初の k+2 項を求めれば多項式補間できる  O(k log(mod))
    s, Y = 0, [0]  # 第 0 項は 0
    for x in range(k+1):
        s += pow(x, k, mod)
        Y.append(s)
    return fast_lagrange_interpolation(Y, n, mod)


class RollingHash:
    # 未検証
    # 参考1:  http://tjkendev.github.io/procon-library/python/string/rolling_hash.html
    # 参考2:  https://ei1333.github.io/algorithm/rolling-hash.html
    BASE = 1000
    MOD = 1111111111111111111  # ≒10**18 素数  # 10**9くらいの素数2つ使うのとどちらが速い？
    def __init__(self, s):
        self.s = s
        self.n = n = len(s)
        BASE = RollingHash.BASE
        MOD = RollingHash.MOD
        self.h = h = [0]*(n+1)
        for i in range(n):
            h[i+1] = (h[i] * BASE + ord(s[i])) % MOD

    def get(self, l, r):  # [l, r)
        MOD = RollingHash.MOD
        return (self.h[r] - self.h[l]*pow(RollingHash.BASE, r-l, MOD)) % MOD

    @classmethod
    def connect(cls, h1, h2, h2len):
        return (h1 * pow(cls.BASE, h2len, cls.MOD) + h2) % cls.MOD

    def lcp(self, h2, l1, r1, l2, r2):  # 最長共通接頭辞
        # 区間の長さ N に対して O(logN)  # h2 は RollingHash オブジェクト
        # 自身の [l1, r1) と h2 の [l2, r2) の最長共通接頭辞の長さを返す
        length = min(r1-l1, r2-l2)
        ok, ng = 0, length+1
        while ng - ok > 1:
            c = ok + ng >> 1
            if self.get(l1, l1+c) == h2.get(l2, l2+c):
                ok = c
            else:
                ng = c
        return ok


def convolve(A, B):
    # 畳み込み (Numpy)  # 要素は整数
    # 3 つ以上の場合は一度にやった方がいい
    import numpy as np
    dtype = np.int64  # np.float128 は windows では動かない？
    fft, ifft = np.fft.rfft, np.fft.irfft
    a, b = len(A), len(B)
    if a==b==1:
        return np.array([A[0]*B[0]])
    n = a + b - 1  # 返り値のリストの長さ
    k = 1 << (n-1).bit_length()  # n 以上の最小の 2 冪
    AB = np.zeros((2, k), dtype=dtype)
    AB[0, :a] = A
    AB[1, :b] = B
    return np.rint(ifft(fft(AB[0]) * fft(AB[1]))).astype(np.int64)[:n]

def garner(A, M, mod):
    # Garner のアルゴリズム (NumPy)
    # 参考: https://math314.hateblo.jp/entry/2015/05/07/014908
    M.append(mod)
    coffs = [1] * len(M)
    constants = np.zeros((len(M),)+A[0].shape, dtype=np.int64)
    for i, (a, m) in enumerate(zip(A, M[:-1])):
        v = (a - constants[i]) * pow(coffs[i], m-2, m) % m
        for j, mm in enumerate(M[i+1:], i+1):
            constants[j] = (constants[j] + coffs[j] * v) % mm
            coffs[j] = coffs[j] * m % mm
    return constants[-1]

def convolve_mod(A, B, mod=10**9+7):
    # 任意 mod 畳み込み (NumPy)
    # 検証1: 注文の多い高橋商店 (TLE): https://atcoder.jp/contests/arc028/submissions/7467522
    # 検証2: [yosupo] Convolution (mod 1,000,000,007): https://judge.yosupo.jp/submission/12504
    
    #mods = [1000003, 1000033, 1000037, 1000039]  # 要素数が 10**3 以下の場合（誤差 6*2+3=15<16  復元 6*4=24>21=9*2+3）
    #mods = [100003, 100019, 100043, 100049, 100057]  # 要素数が10**5 以下の場合（誤差 5*2+5=15<16  復元 5*5=25>23=9*2+5）
    mods = [63097, 63103, 63113, 63127, 63131]  # 要素数が 5*10**5 以下の場合（誤差 4.8*2+5.7=15.3<16  復元 4.8*5=24>23.7=9*2+5.7  10**4.8=63096  10**5.7=501187）
    
    mods_np = np.array(mods, dtype=np.int32)
    fft, ifft = np.fft.rfft, np.fft.irfft
    a, b = len(A), len(B)
    if a == b == 1:
        return np.array([A[0] * B[0]]) % mod
    n = a + b - 1  # 畳み込みの結果の長さ
    k = 1 << (n - 1).bit_length()  # n 以上の最小の 2 冪
    AB = np.zeros((2, len(mods), k), dtype=np.int64)  # ここの dtype は fft 後の dtype に関係しない
    AB[0, :, :a] = A
    AB[1, :, :b] = B
    AB[:, :, :] %= mods_np[:, None]
    C = ifft(fft(AB[0]) * fft(AB[1]))[:, :n]
    C = ((C + 0.5) % mods_np[:, None]).astype(np.int64)
    return garner(C, mods, mod)


def scc(E, n_vertex):
    # 強連結成分分解 (NumPy, SciPy)  # E は [[a1, b1], [a2, b2], ... ] の形
    # 返り値は 強連結成分の数 と 各頂点がどの強連結成分に属しているか
    # numpy いらないのは https://tjkendev.github.io/procon-library/python/graph/scc.html
    import numpy as np
    from scipy.sparse import csr_matrix, csgraph
    A, B = np.array(E).T
    graph = csr_matrix((np.ones(len(E)), (A, B)), (n_vertex, n_vertex))
    n_components, labels = csgraph.connected_components(graph, connection='strong')
    return n_components, labels


def distribute(n, person, min, max, mode="even"):
    # n 個を person 人に分配する
    # 返り値は [[a (個), a 個もらう人数], ...]
    # 分配できないときは None を返す
    if person==0 and n==0:
        return []
    elif not min*person <= n <= max*person:
        return None
    elif mode=="even":
        q, m = divmod(n, person)
        if m==0:
            return [[q, person]]
        else:
            return [[q, person-m], [q+1, m]]
    elif mode=="greedy":
        if max==min:
            return [[max, person]]
        n -= min * person
        q, m = divmod(n, max-min)
        if m==0:
            return [[min, person-q], [max, q]]
        else:
            return [[min, person-1-q], [min+m, 1], [max, q]]
    else:
        raise ValueError("'mode' must be 'even' or 'greedy'.")


def xorshift(seed=123456789):  # 31 bit xorshift
    y = seed
    def randint(a, b):  # 閉区間
        nonlocal y
        y ^= (y & 0xffffff) << 7
        y ^= y >> 12
        return y % (b-a+1) + a
    return randint


"""
# 重み付き UnionFind https://atcoder.jp/contests/code-festival-2016-quala/submissions/8336387
# Trie https://atcoder.jp/contests/code-festival-2016-qualb/submissions/8335110
# 全方位木 dp https://atcoder.jp/contests/yahoo-procon2019-final-open/submissions/8664902
# 平方分割
#  I hate Shortest Path Problem https://atcoder.jp/contests/abc177/submissions/16384352
#  Replace Digits https://atcoder.jp/contests/abl/submissions/17049200
#  天下一数列にクエリを投げます https://atcoder.jp/contests/tenka1-2016-qualb/submissions/14415635
#  Range Affine Range Sum https://atcoder.jp/contests/practice2/submissions/17100361


A = csgraph.dijkstra(X, indices=0)

zip(*[iter(Ans)]*3)  # 3 個ずつ
zip(*[iter(map(int, sys.stdin.read().split()))]*4):

https://github.com/Lgeu/snippet/
import sys
input = sys.stdin.readline
C = np.frombuffer(buf.read(), dtype="S1").reshape(H, W+1)[:, :-1].T
"""

