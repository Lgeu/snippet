class SegTree:
    def __init__(self, op, e, n):
        # n は要素数または iterable
        # O(n)
        pass

    def set(self, p, x):
        # O(log(n))
        pass

    def get(self, p):
        # O(1)
        pass

    def prod(self, l, r):
        # O(log(n))
        pass

    def all_prod(self):
        # O(1)
        pass

    def max_right(self, l, f):
        # O(log(n))
        pass

    def max_left(self, r, f):
        # O(log(n))
        pass

    def to_list(self):
        # O(n)
        pass

def test_segtree():
    from atcoder import SegTree
    
    input = sys.stdin.buffer.readline
    N, Q = map(int, input().split())
    A = map(int, input().split())

    op = lambda a, b: a if a > b else b
    e = -1
    seg = SegTree(op, e, A)

    Ans = []
    m = map(int, sys.stdin.buffer.read().split())
    for t, x, v in zip(m, m, m):
        x -= 1
        if t == 1:
            seg.set(x, v)
        elif t == 2:
            ans = seg.prod(x, v)
            Ans.append(ans)
        else:
            ans = seg.max_right(x, lambda a: a < v) + 1
            Ans.append(ans)
    if Ans:
        print("\n".join(map(str, Ans)))


class LazySegTree:
    def __init__(self, op, e, mapping, composition, identity, n):
        # n は要素数または iterable
        # O(n)
        pass

    def set(self, p, x):
        # O(log(n))
        pass

    def get(self, p):
        # O(log(n))
        pass

    def prod(self, l, r):
        # O(log(n))
        pass

    def prod_getitem(self, l, r, idx):  # original
        # O(log(n))
        pass
    
    def all_prod(self):
        # O(1)
        pass

    def apply(self, l, r, x=None):
        # O(log(n)
        # 2 引数で呼び出した場合は p, x で、 p の 1 箇所にのみ適用
        pass

    def max_right(self, l, f):
        # O(log(n))
        pass

    def max_left(self, r, f):
        # O(log(n))
        pass

    def to_list(self):
        # O(n) のはず
        pass



