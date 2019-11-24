from bisect import bisect_left, bisect_right, insort_right
class SquareSkipList:
    # SkipList の層数を 2 にした感じの何か
    # std::multiset の代用になる
    # 検証1 (add, pop) データ構造: https://atcoder.jp/contests/arc033/submissions/7488620
    # 検証2 (init, add, remove, search_higher_equal) Exclusive OR Queries: https://atcoder.jp/contests/cpsco2019-s1/submissions/7488225
    # 検証3 (add, search_higher, search_lower) Second Sum: https://atcoder.jp/contests/abc140/submissions/7488469
    # 検証4 (add, __getitem__) [CF] Optimal Subsequences (Hard Version): https://codeforces.com/contest/1261/submission/65643461
    def __init__(self, values=None, sorted_=False, square=1000, seed=42):
        # values: 初期値のリスト
        # sorted_: 初期値がソート済みであるか
        # square: 最大データ数の平方根
        # seed: 乱数のシード
        inf = float("inf")
        self.square = square
        if values is None:
            self.rand_y = seed
            self.layer1 = [inf]
            self.layer0 = [[]]
        else:
            self.layer1 = layer1 = []
            self.layer0 = layer0 = []
            if not sorted_:
                values.sort()
            y = seed
            l0 = []
            for v in values:
                y ^= (y & 0x7ffff) << 13
                y ^= y >> 17
                y ^= (y & 0x7ffffff) << 5
                if y % square == 0:
                    layer0.append(l0)
                    l0 = []
                    layer1.append(v)
                else:
                    l0.append(v)
            layer1.append(inf)
            layer0.append(l0)
            self.rand_y = y

    def add(self, x):  # 要素の追加  # O(sqrt(n))
        # xorshift
        y = self.rand_y
        y ^= (y & 0x7ffff) << 13
        y ^= y >> 17
        y ^= (y & 0x7ffffff) << 5
        self.rand_y = y

        if y % self.square == 0:
            layer1, layer0 = self.layer1, self.layer0
            idx1 = bisect_right(layer1, x)
            layer1.insert(idx1, x)
            layer0_idx1 = layer0[idx1]
            idx0 = bisect_right(layer0_idx1, x)
            layer0.insert(idx1+1, layer0_idx1[idx0:])  # layer0 は dict で管理した方が良いかもしれない  # dict 微妙だった
            del layer0_idx1[idx0:]
        else:
            idx1 = bisect_right(self.layer1, x)
            insort_right(self.layer0[idx1], x)

    def remove(self, x):  # 要素の削除  # O(sqrt(n))
        # x が存在しない場合、x 以上の最小の要素が削除される
        idx1 = bisect_left(self.layer1, x)
        layer0_idx1 = self.layer0[idx1]
        idx0 = bisect_left(layer0_idx1, x)
        if idx0 == len(layer0_idx1):
            del self.layer1[idx1]
            self.layer0[idx1] += self.layer0.pop(idx1+1)
        else:
            del layer0_idx1[idx0]

    def search_higher_equal(self, x):  # x 以上の最小の値を返す  O(log(n))
        idx1 = bisect_left(self.layer1, x)
        layer0_idx1 = self.layer0[idx1]
        idx0 = bisect_left(layer0_idx1, x)
        if idx0 == len(layer0_idx1):
            return self.layer1[idx1]
        return layer0_idx1[idx0]

    def search_higher(self, x):  # x を超える最小の値を返す  O(log(n))
        idx1 = bisect_right(self.layer1, x)
        layer0_idx1 = self.layer0[idx1]
        idx0 = bisect_right(layer0_idx1, x)
        if idx0 == len(layer0_idx1):
            return self.layer1[idx1]
        return layer0_idx1[idx0]

    def search_lower(self, x):  # x 未満の最大の値を返す  O(log(n))
        idx1 = bisect_left(self.layer1, x)
        layer0_idx1 = self.layer0[idx1]
        idx0 = bisect_left(layer0_idx1, x)
        if idx0 == 0:  # layer0_idx1 が空の場合とすべて x 以上の場合
            return self.layer1[idx1-1]
        return layer0_idx1[idx0-1]

    def pop(self, idx):
        # 小さい方から idx 番目の要素を削除してその要素を返す（0-indexed）
        # O(sqrt(n))
        # for を回すので重め、使うなら square パラメータを大きめにするべき
        layer0 = self.layer0
        s = -1
        for i, l0 in enumerate(layer0):
            s += len(l0) + 1
            if s >= idx:
                break
        if s==idx:
            layer0[i] += layer0.pop(i+1)
            return self.layer1.pop(i)
        else:
            return layer0[i].pop(idx-s)
        
    def __getitem__(self, item):
        # 小さい方から idx 番目の要素を返す  O(sqrt(N))
        layer0 = self.layer0
        s = -1
        for i, l0 in enumerate(layer0):
            s += len(l0) + 1
            if s >= item:
                break
        if s == item:
            return self.layer1[i]
        else:
            return layer0[i][item-s]

    def print(self):
        print(self.layer1)
        print(self.layer0)

