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


class SegTreeIndex(object):
    # 区間の最小 or 最大値とその index を取得
    # 未検証
    __slots__ = ["elem_size", "tree", "default", "op", "index", "op2"]
    def __init__(self, a: list, default=float("inf"), op=min, op2=lambda a,b:a>b):
        # 同じ場合最左のインデックスを返す
        # 最大値・最左 -> -float("inf"), max, lambda a,b:a<=b
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
        tree, result, op, op2, index = self.tree, self.default, self.op, self.op2, self.index
        idx = -1
        while l < r:
            if l & 1:
                v1, v2 = result, tree[l]
                result = op(v1, v2)
                if op2(v1, v2)==1:
                    idx = index[l]
                l += 1
            if r & 1:
                r -= 1
                v1, v2 = tree[r], result
                result = op(v1, v2)
                if op2(v1, v2)==0:
                    idx = index[l]
            l, r = l >> 1, r >> 1
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
