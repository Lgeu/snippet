class Bit:
    # 参考1: http://hos.ac/slides/20140319_bit.pdf
    # 参考2: https://atcoder.jp/contests/arc046/submissions/6264201
    # 検証: https://atcoder.jp/contests/arc046/submissions/7435621
    # values の 0 番目は使わない
    # len(values) を 2 冪 +1 にすることで二分探索の条件を減らす
    def __init__(self, a):
        if hasattr(a, "__iter__"):
            le = len(a)
            self.n = 1 << le.bit_length()  # le を超える最小の 2 冪
            self.values = values = [0] * (self.n+1)
            values[1:le+1] = a[:]
            for i in range(1, self.n):
                values[i + (i & -i)] += values[i]
        elif isinstance(a, int):
            self.n = 1 << a.bit_length()
            self.values = [0] * (self.n+1)
        else:
            raise TypeError
 
    def add(self, i, val):
        n, values = self.n, self.values
        while i <= n:
            values[i] += val
            i += i & -i
 
    def sum(self, i):  # (0, i]
        values = self.values
        res = 0
        while i > 0:
            res += values[i]
            i -= i & -i
        return res
 
    def bisect_left(self, v):  # self.sum(i) が v 以上になる最小の i
        n, values = self.n, self.values
        if v > values[n]:
            return None
        i, step = 0, n>>1
        while step:
            if values[i+step] < v:
                i += step
                v -= values[i]
            step >>= 1
        return i + 1
