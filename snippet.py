# 拡張ユークリッド互除法
# ax + by = gcd(a,b)の最小整数解を返す
# 最大公約数はg
def egcd(a, b):
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

# nCr mod m
# rがn/2に近いと非常に重くなる
def combination(n, r, mod=10**9+7):
    n1, r = n+1, min(r, n-r)
    numer = denom = 1
    for i in range(1, r+1):
        numer = numer * (n1-i) % mod
        denom = denom * i % mod
    return numer * pow(denom, mod-2, mod) % mod

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
        if isPrime[i]:
            for j in range(i * i, n + 1, i):
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


def miller_rabin(n, t=10):
    # http://tjkendev.github.io/procon-library/python/prime/probabilistic.html
    # 確率的素数判定（ミラーラビン素数判定法）
    # 素数なら確実に 1 を返す、合成数なら確率的に 0 を返す
    # 1 が返ったなら恐らく素数で、0 が返ったなら確実に合成数である
    # t が多いほど精度が良い
    import random
    if n==2: return 1
    if n==1 or n&1==0: return 0
    d = n-1
    d //= d & -d
    for _ in range(t):
        a = random.randint(1, n-1)
        t = d
        y = pow(a, t, n)
        while t!=n-1 and y!=1 and y!=n-1:
            y = y**2 % n
            t <<= 1
        if y!=n-1 and t&1==0: return 0
    return 1

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
"""
    0-indexed
    # 使用例
    bit = Bit(10)  # 要素数
    bit.add(2, 10)
    print(bit.sum(5))  # 10
"""
class Bit:
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


#E = defaultdict(lambda: defaultdict(lambda: float("inf")))
from collections import defaultdict
import heapq
class Dijkstra:
    # 計算量 O((E+V)logV)

    # adjはdefaultdictのリスト
    def dijkstra(self, adj, start, goal=None):

        num = len(adj)  # グラフのノード数
        self.dist = [float('inf') for i in range(num)]  # 始点から各頂点までの最短距離を格納する
        self.prev = [float('inf') for i in range(num)]  # 最短経路における，その頂点の前の頂点のIDを格納する

        self.dist[start] = 0
        q = [(0, start)]  # プライオリティキュー．各要素は，(startからある頂点vまでの仮の距離, 頂点vのID)からなるタプル

        while len(q) != 0:
            prov_cost, src = heapq.heappop(q)  # pop

            # プライオリティキューに格納されている最短距離が，現在計算できている最短距離より大きければ，distの更新をする必要はない
            if self.dist[src] < prov_cost:
                continue

            # 探索で辺を見つける場合ここに書く


            # 他の頂点の探索
            for dest, cost in adj[src].items():
                if self.dist[dest] > self.dist[src] + cost:
                    self.dist[dest] = self.dist[src] + cost  # distの更新
                    heapq.heappush(q, (self.dist[dest], dest))  # キューに新たな仮の距離の情報をpush
                    self.prev[dest] = src  # 前の頂点を記録

        if goal is not None:
            return self.get_path(goal, self.prev)
        else:
            return self.dist

    def get_path(self, goal, prev):
        path = [goal]  # 最短経路
        dest = goal

        # 終点から最短経路を逆順に辿る
        while prev[dest] != float('inf'):
            path.append(prev[dest])
            dest = prev[dest]

        # 経路をreverseして出力
        return list(reversed(path))


# unionfind
class Uf:
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

        if u == v: return

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
# 高速ゼータ変換

# 自身を含む集合を全て挙げる方
N = 3
f = [{i} for i in range(1<<N)]
for i in range(N):
    for j in range(1<<N):
        if not (j & 1<<i):
            f[j] |= f[j | (1<<i)]  # 総和は +=  # -=にすると逆変換になる
print(f)

# 部分集合をすべて挙げる方
f = [{i} for i in range(1<<N)]
for i in range(N):
    for j in range(1<<N):
        if j & 1<<i:
            f[j] |= f[j ^ (1<<i)]
print(f)
"""


# https://atcoder.jp/contests/abc014/submissions/3935971
class SegmentTree(object):
    __slots__ = ["elem_size", "tree", "default", "op"]

    def __init__(self, a: list, default: int, op):
        from math import ceil, log
        real_size = len(a)
        self.elem_size = elem_size = 1 << ceil(log(real_size, 2))
        self.tree = tree = [default] * (elem_size * 2)
        tree[elem_size:elem_size + real_size] = a
        self.default = default
        self.op = op

        for i in range(elem_size - 1, 0, -1):
            tree[i] = op(tree[i << 1], tree[(i << 1) + 1])

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
        self.tree[k] = value
        self.update(k)

    def update(self, i: int) -> None:
        op, tree = self.op, self.tree
        while i > 1:
            i >>= 1
            tree[i] = op(tree[i << 1], tree[(i << 1) + 1])
"""
C = [int(input()) for _ in range(N)]

idx = [0] * N
for i, c in enumerate(C):
    idx[c-1] = i

seg = SegmentTree([0]*(N+1), 0, min)
for i in range(N):
    idx_ = idx[i]
    seg.set_value(idx_, seg.get_value(0, idx_)-1)
print(seg.get_value(0, N+1)+N)
"""

# 最長回文
def man(S):
    i = 0
    j = 0
    n = len(S)
    R = [0]*n
    while i < n:
        while i-j >= 0 and i+j < n and S[i-j] == S[i+j]:
            j+=1
        R[i] = j
        k = 1
        while i-k >= 0 and i+k < n and k+R[i-k] < j:
            R[i+k] = R[i-k]
            k += 1
        i += k
        j -= k
    return R

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


# 凸包 Monotone Chain O(nlogn)
# 参考: https://matsu7874.hatenablog.com/entry/2018/12/17/025713
# 複素数版は https://atcoder.jp/contests/abc139/submissions/7301049
def get_convex_hull(points):
    def det(p, q):
        return p[0] * q[1] - p[1] * q[0]
    def sub(p, q):
        return (p[0] - q[0], p[1] - q[1])
    points.sort()
    ch = []
    for p in points:
        while len(ch) > 1:
            v_cur = sub(ch[-1], ch[-2])
            v_new = sub(p, ch[-2])
            if det(v_cur, v_new) > 0:
                break
            ch.pop()
        ch.append(p)
    t = len(ch)
    for p in points[-2::-1]:
        while len(ch) > t:
            v_cur = sub(ch[-1], ch[-2])
            v_new = sub(p, ch[-2])
            if det(v_cur, v_new) > 0:
                break
            ch.pop()
        ch.append(p)
    return ch[:-1]


# 線分 AB と CD の交差判定
def cross(x1, y1, x2, y2, x3, y3, x4, y4):
    def f(x, y, x1, y1, x2, y2):  # 直線上にあるとき 0 になる
        return (x1-x2)*(y-y1)+(y1-y2)*(x1-x)

    # 点 C と点 D が直線 AB の異なる側にある
    b1 = f(x3, y3, x1, y1, x2, y2) * f(x4, y4, x1, y1, x2, y2) < 0

    # 点 A と点 B が直線 CD の異なる側にある
    b2 = f(x1, y1, x3, y3, x4, y4) * f(x2, y2, x3, y3, x4, y4) < 0

    return b1 and b2


def intersection(circle, polygon):
    # circle: (x, y, r)
    # polygon: [(x1, y1), (x2, y2), ...]
    # 円と多角形の共通部分の面積
    # 多角形の点が反時計回りで与えられれば正の値、時計回りなら負の値を返す
    from math import acos, hypot, isclose, sqrt
    def cross(v1, v2):  # 外積
        x1, y1 = v1
        x2, y2 = v2
        return x1 * y2 - x2 * y1

    def dot(v1, v2):  # 内積
        x1, y1 = v1
        x2, y2 = v2
        return x1 * x2 + y1 * y2

    def seg_intersection(circle, seg):
        # 円と線分の交点（円の中心が原点でない場合は未検証）
        x0, y0, r = circle
        p1, p2 = seg
        x1, y1 = p1
        x2, y2 = p2

        p1p2 = (x2 - x1) ** 2 + (y2 - y1) ** 2
        op1 = (x1 - x0) ** 2 + (y1 - y0) ** 2
        rr = r * r
        dp = dot((x1 - x0, y1 - y0), (x2 - x1, y2 - y1))

        d = dp * dp - p1p2 * (op1 - rr)
        ps = []

        if isclose(d, 0.0, abs_tol=1e-9):
            t = -dp / p1p2
            if ge(t, 0.0) and le(t, 1.0):
                ps.append((x1 + t * (x2 - x1), y1 + t * (y2 - y1)))
        elif d > 0.0:
            t1 = (-dp - sqrt(d)) / p1p2
            if ge(t1, 0.0) and le(t1, 1.0):
                ps.append((x1 + t1 * (x2 - x1), y1 + t1 * (y2 - y1)))
            t2 = (-dp + sqrt(d)) / p1p2
            if ge(t2, 0.0) and le(t2, 1.0):
                ps.append((x1 + t2 * (x2 - x1), y1 + t2 * (y2 - y1)))

        # assert all(isclose(r, hypot(x, y)) for x, y in ps)
        return ps

    def le(f1, f2):  # less equal
        return f1 < f2 or isclose(f1, f2, abs_tol=1e-9)

    def ge(f1, f2):  # greater equal
        return f1 > f2 or isclose(f1, f2, abs_tol=1e-9)

    x, y, r = circle
    polygon = [(xp-x, yp-y) for xp, yp in polygon]
    area = 0.0
    for p1, p2 in zip(polygon, polygon[1:] + [polygon[0]]):
        ps = seg_intersection((0, 0, r), (p1, p2))
        for pp1, pp2 in zip([p1] + ps, ps + [p2]):
            c = cross(pp1, pp2)  # pp1 と pp2 の位置関係によって正負が変わる
            if c == 0:  # pp1, pp2, 原点が同一直線上にある場合
                continue
            d1 = hypot(*pp1)
            d2 = hypot(*pp2)
            if le(d1, r) and le(d2, r):
                area += c / 2  # pp1, pp2, 原点を結んだ三角形の面積
            else:
                t = acos(dot(pp1, pp2) / (d1 * d2))  # pp1-原点とpp2-原点の成す角
                sign = 1.0 if c >= 0 else -1.0
                area += sign * r * r * t / 2  # 扇形の面積
    return area

def lis(A: list):  # 最長増加部分列
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

class Lca:  # 最近共通祖先
    def __init__(self, E, root):
        import sys
        sys.setrecursionlimit(500000)
        self.root = root
        self.E = E  # V<V>
        self.n = len(E)  # 頂点数
        self.logn = 1  # n < 1<<logn  ぴったりはだめ
        while self.n >= (1<<self.logn):
            self.logn += 1

        # parent[n][v] = ノード v から 1<<n 個親をたどったノード
        self.parent = [[-1]*self.n for _ in range(self.logn)]

        self.depth = [0] * self.n
        self.dfs(root, -1, 0)
        for k in range(self.logn-1):
            for v in range(self.n):
                p_ = self.parent[k][v]
                if p_ >= 0:
                    self.parent[k+1][v] = self.parent[k][p_]

    def dfs(self, v, p, dep):
        # ノード番号、親のノード番号、深さ
        self.parent[0][v] = p
        self.depth[v] = dep
        for e in self.E[v]:
            if e != p:
                self.dfs(e, v, dep+1)

    def get(self, u, v):
        if self.depth[u] > self.depth[v]:
            u, v = v, u  # self.depth[u] <= self.depth[v]
        dep_diff = self.depth[v]-self.depth[u]
        for k in range(self.logn):
            if dep_diff >> k & 1:
                v = self.parent[k][v]
        if u==v:
            return u
        for k in range(self.logn-1, -1, -1):
            if self.parent[k][u] != self.parent[k][v]:
                u = self.parent[k][u]
                v = self.parent[k][v]
        return self.parent[0][u]

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
    # 畳み込み  # 要素は整数
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

def scc(E, n_vertex):
    # 強連結成分分解  # E は [[a1, b1], [a2, b2], ... ] の形
    # 返り値は 強連結成分の数 と 各頂点がどの強連結成分に属しているか
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

"""
A = csgraph.dijkstra(X, indices=0)

https://github.com/Lgeu/snippet/
import sys
input = sys.stdin.readline
def input():
    return sys.stdin.readline()[:-1]

from functools import lru_cache
@lru_cache(maxsize=None)  # メモ化再帰したい関数の前につける

import sys
sys.setrecursionlimit(500000)

N = int(input())
N, K = map(int, input().split())
L = [int(input()) for _ in range(N)]
A = list(map(int, input().split()))
S = [list(map(int, input().split())) for _ in range(H)]
"""

