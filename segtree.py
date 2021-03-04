class SegmentTree(object):
    # https://atcoder.jp/contests/abc014/submissions/3935971
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

# モノイドの元として (index, value) を使うと index も取得できるようになる
seg = SegmentTree(list(enumerate(A)),
                  (-1, float("inf")),
                  lambda x, y: x if x[1]<y[1] else y)


class SegTreeIndex(object):
    # 区間の最小 or 最大値とその index を取得
    # 検証1: https://yukicoder.me/submissions/376308
    # 検証2: https://atcoder.jp/contests/abc146/submissions/8634746
    __slots__ = ["elem_size", "tree", "default", "op", "index", "op2"]
    def __init__(self, a: list, default=float("inf"), op=lambda a,b: b if a>b else a, op2=lambda a,b:a>b):
        # 同じ場合最左のインデックスを返す
        # 最小値・最左: default=float("inf"), op=lambda a,b: b if a>b else a, op2=lambda a,b:a>b
        # 最大値・最左: -float("inf"), max, lambda a,b:a<=b
        from math import ceil, log
        real_size = len(a)
        self.elem_size = elem_size = 1 << ceil(log(real_size, 2))
        self.tree = tree = [default] * (elem_size * 2)
        self.index = index = [0] * (elem_size * 2)
        tree[elem_size:elem_size + real_size] = a
        index[elem_size:elem_size + real_size] = list(range(real_size))
        self.default = default
        self.op = op
        self.op2 = op2
        for i in range(elem_size-1, 0, -1):
            v1, v2 = tree[i<<1], tree[(i<<1)+1]
            tree[i] = op(v1, v2)
            index[i] = index[(i<<1) + op2(v1, v2)]

    def get_value(self, x: int, y: int) -> tuple:  # 半開区間
        l, r = x + self.elem_size, y + self.elem_size
        tree, op, op2, index = self.tree, self.op, self.op2, self.index
        result_l = result_r = self.default
        idx_l = idx_r = -1
        while l < r:
            if l & 1:
                v1, v2 = result_l, tree[l]
                result_l = op(v1, v2)
                if op2(v1, v2)==1:
                    idx_l = index[l]
                l += 1
            if r & 1:
                r -= 1
                v1, v2 = tree[r], result_r
                result_r = op(v1, v2)
                if op2(v1, v2)==0:
                    idx_r = index[r]
            l, r = l >> 1, r >> 1
        result = op(result_l, result_r)
        idx = idx_r if op2(result_l, result_r) else idx_l
        return result, idx

    def set_value(self, i: int, value: int) -> None:
        k = self.elem_size + i
        self.tree[k] = value
        self.update(k)

    def update(self, i: int) -> None:
        op, tree, index, op2 = self.op, self.tree, self.index, self.op2
        while i > 1:
            i >>= 1
            v1, v2 = tree[i<<1], tree[(i<<1)+1]
            tree[i] = op(v1, v2)
            index[i] = index[(i<<1) + op2(v1, v2)]


class SegTree(object):
    # 区間の中で v 以下の値のうち最も左にある値と index を取得
    # 普通のセグ木に get_threshold_left と get_threshold_left_all を加えただけ
    # 検証1: https://atcoder.jp/contests/arc038/submissions/6933949 (全区間のみ)
    # 検証2: https://atcoder.jp/contests/arc046/submissions/7430924 (全区間のみ)
    # 抽象化したい
    __slots__ = ["elem_size", "tree", "default", "op"]
    def __init__(self, a: list, default=float("inf"), op=min):
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

    def get_threshold_left(self, x, y, v):
        # 区間 [x, y) 内で一番左の v 以下の値
        tree, result, op, elem_size = self.tree, self.default, self.op, self.elem_size
        l, r = x + elem_size, y + elem_size
        idx_left = idx_right = -1  # 内部 index
        while l < r:
            if l & 1:
                result = op(tree[l], result)
                if idx_left == -1 and tree[l] <= v:
                    idx_left = l
                l += 1
            if r & 1:
                r -= 1
                result = op(tree[r], result)
                if tree[r] <= v:
                    idx_right = r
            l, r = l >> 1, r >> 1
        if idx_left==idx_right==-1:
            return -1, -1
        idx = idx_left if idx_left!=-1 else idx_right
        while idx < elem_size:
            idx <<= 1
            if tree[idx] > v:
                idx += 1
        return tree[idx], idx-elem_size

    def get_threshold_left_all(self, v):
        # 全区間で一番左の v 以下の値
        tree, op, elem_size = self.tree, self.op, self.elem_size
        if tree[1] > v:
            return -1, -1
        idx = 1
        while idx < elem_size:
            idx <<= 1
            if tree[idx] > v:
                idx += 1
        return tree[idx], idx-elem_size

    def set_value(self, i: int, value: int) -> None:
        k = self.elem_size + i
        self.tree[k] = value
        self.update(k)

    def update(self, i: int) -> None:
        op, tree = self.op, self.tree
        while i > 1:
            i >>= 1
            tree[i] = op(tree[i << 1], tree[(i << 1) + 1])


class SparseTable:
    def __init__(self, values, op=min, zero_element=float("inf")):  # O(nlogn * (op の計算量))
        self.n = n = len(values)
        self.table = table = [values]
        self.op = op
        self.zero_element = zero_element
        for d in range(n.bit_length()-1):
            table.append([op(v1, v2) for v1, v2 in zip(table[-1], table[-1][1<<d:])])
 
    def get_value(self, l, r):  # 半開区間  # O(op の計算量)
        bl_m1 = (r-l).bit_length() - 1
        t = self.table[bl_m1]
        return self.op(t[l], t[r-(1<<bl_m1)])
 
    def arc023d(self, l, g):  # op(values[l:r]) が g 以上である最大の r を求める  O(logn * (op の計算量))
        bl_m1, op = (self.n-l).bit_length()-1, self.op
        g_prov, r, step = self.zero_element, l, 1<<bl_m1
        for t in self.table[bl_m1::-1]:
            if r < len(t):
                g_next = op(g_prov, t[r])
                if g_next >= g:
                    g_prov = g_next
                    r += step
            step >>= 1
        return r


class Rmq:
    # 平方分割
    # 値を変更すると元のリストの値も書き換わる
    # 検証: http://judge.u-aizu.ac.jp/onlinejudge/review.jsp?rid=3990681
    def __init__(self, a, sqrt_n=150, inf=(1<<31)-1):
        self.sqrt_n = sqrt_n
        if hasattr(a, "__iter__"):
            from itertools import zip_longest
            self.n = len(a)
            self.layer0 = [min(values) for values in zip_longest(*[iter(a)]*sqrt_n, fillvalue=inf)]
            self.layer1 = a
        elif isinstance(a, int):
            self.n = a
            self.layer0 = [inf] * ((a - 1) // sqrt_n + 1)
            self.layer1 = [inf] * a
        else:
            raise TypeError

    def get_min(self, l, r):
        sqrt_n = self.sqrt_n
        parent_l, parent_r = l//sqrt_n+1, (r-1)//sqrt_n
        if parent_l < parent_r:
            return min(min(self.layer0[parent_l:parent_r]),
                       min(self.layer1[l:parent_l*sqrt_n]),
                       min(self.layer1[parent_r*sqrt_n:r]))
        else:
            return min(self.layer1[l:r])

    def set_value(self, idx, val):
        self.layer1[idx] = val
        idx0 = idx // self.sqrt_n
        idx1 = idx0 * self.sqrt_n
        self.layer0[idx0] = min(self.layer1[idx1:idx1+self.sqrt_n])

    def chmin(self, idx, val):
        if self.layer1[idx] > val:
            self.layer1[idx] = val
            idx //= self.sqrt_n
            self.layer0[idx] = min(self.layer0[idx], val)

    def debug(self):
        print("layer0=", self.layer0)
        print("layer1=", self.layer1)

    def __getitem__(self, item):
        return self.layer1[item]

    def __setitem__(self, key, value):
        self.set_value(key, value)

# https://atcoder.jp/contests/nikkei2019-2-qual/submissions/8434117 もある
