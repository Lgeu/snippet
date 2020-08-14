class Lca:  # 最近共通祖先（ダブリング）
    # HL 分解での LCA を書いたのでボツ
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

def get_convex_hull(points):  # 凸包
    # 複素数のものを使っていきたいのでボツ
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

