# 枝刈り探索（分枝限定法）
# 検証: https://atcoder.jp/contests/abc032/submissions/8197125
 
class Knapsack:
    def __init__(self, VW):
        self.VW = VW
        self.VW.sort(key=lambda vw: vw[0] / vw[1], reverse=True)
        self.n = len(VW)
 
    def solve(self, capacity, ok=0):
        self.ok = ok
        self.capacity = capacity
        return self._dfs(0, 0, 0)
 
    def _dfs(self, i, v_now, w_now):
        if i==self.n:
            self.ok = max(self.ok, v_now)
            return v_now
        ng, f = self._solve_relaxation(i, self.capacity-w_now)
        ng += v_now
        if f:
            self.ok = max(self.ok, ng)
            return ng
        if ng < self.ok:
            return -float("inf")
        res = -float("inf")
        v, w = self.VW[i]
        if w_now + w <= self.capacity:
            res = max(res, self._dfs(i+1, v_now + v, w_now + w))
        res = max(res, self._dfs(i+1, v_now, w_now))
        return res
 
    def _solve_relaxation(self, i, capacity):
        res = 0
        f = True
        for v, w in self.VW[i:]:
            if capacity == 0:
                break
            if w <= capacity:
                capacity -= w
                res += v
            else:
                f = False
                res += v * (capacity / w)
                break
        return res, f
 
def main():
    N, W = map(int, input().split())
    VW = [list(map(int, input().split())) for _ in range(N)]
    knapsack = Knapsack(VW)
    print(knapsack.solve(W))
