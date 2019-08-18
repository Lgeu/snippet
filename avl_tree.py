class AvlTree:  # std::set
    def __init__(self, values=None, sorted_=False, n=0):
        # values: 初期値のリスト
        # sorted_: 初期値がソート済みであるか
        # n: add メソッドを使う回数の最大値

        # sorted_==True であれば、初期値の数の線形時間で木を構築する
        # 値を追加するときは必ず n を設定する
        if values is None:
            self.left = [-1] * (n + 1)
            self.right = [-1] * (n + 1)
            self.values = [-float("inf")]
            self.diff = [0] * (n + 1)  # left - right
            self.size_l = [0] * (n + 1)
            self.idx_new_val = 0
        else:
            if not sorted_:
                values.sort()
            len_ = self.idx_new_val = len(values)
            n += len_
            self_left = self.left = [-1] * (n + 1)
            self_right = self.right = [-1] * (n + 1)
            self_values = self.values = [-float("inf")] + values
            self_diff = self.diff = [0] * (n + 1)  # left - right
            self_size_l = self.size_l = [0] * (n + 1)

            st = [[1, len_ + 1, 0]]
            while len(st) > 0:  # dfs っぽく木を構築
                l, r, idx_par = st.pop()  # 半開区間
                c = (l + r) >> 1  # python -> //2  pypy -> >>1
                if self_values[c] < self_values[idx_par]:
                    self_left[idx_par] = c
                else:
                    self_right[idx_par] = c
                siz = r - l
                if siz & -siz == siz != 1:  # 2 冪だったら
                    self_diff[c] = 1
                self_size_l[c] = siz_l = c - l
                if siz_l > 0:
                    st.append([l, c, c])
                    c1 = c + 1
                    if c1 < r:  # 左にノードがなければ右には必ず無いので
                        st.append([c1, r, c])

    def rotate_right(self, idx_par, lr):  # lr: 親の左なら 0
        self_left = self.left
        self_right = self.right
        self_diff = self.diff
        self_size_l = self.size_l

        lr_container = self_right if lr else self_left
        idx = lr_container[idx_par]
        #assert self_diff[idx] == 2
        idx_l = self_left[idx]
        diff_l = self_diff[idx_l]

        if diff_l == -1:  # 複回転
            idx_lr = self_right[idx_l]
            diff_lr = self_diff[idx_lr]
            if diff_lr == 0:
                self_diff[idx] = 0
                self_diff[idx_l] = 0
            elif diff_lr == 1:
                self_diff[idx] = -1
                self_diff[idx_l] = 0
                self_diff[idx_lr] = 0
            else:  # diff_lr == -1
                self_diff[idx] = 0
                self_diff[idx_l] = 1
                self_diff[idx_lr] = 0

            # 部分木の大きさの計算
            self_size_l[idx_lr] += self_size_l[idx_l] + 1
            self_size_l[idx] -= self_size_l[idx_lr] + 1

            # 回転
            self_right[idx_l] = self_left[idx_lr]
            self_left[idx] = self_right[idx_lr]
            self_left[idx_lr] = idx_l
            self_right[idx_lr] = idx
            lr_container[idx_par] = idx_lr

            return 0

        else:  # 単回転
            if diff_l == 0:
                self_diff[idx] = 1
                nb = self_diff[idx_l] = -1
            else:  # diff_l == 1
                self_diff[idx] = 0
                nb = self_diff[idx_l] = 0

            # 部分木の大きさの計算
            self_size_l[idx] -= self_size_l[idx_l] + 1

            # 回転
            self_left[idx] = self_right[idx_l]
            self_right[idx_l] = idx
            lr_container[idx_par] = idx_l

            return nb  # 新しい根の diff を返す

    def rotate_left(self, idx_par, lr):  # lr: 親の左なら 0
        self_left = self.left
        self_right = self.right
        self_diff = self.diff
        self_size_l = self.size_l

        lr_container = self_right if lr else self_left
        idx = lr_container[idx_par]
        #assert self_diff[idx] == -2
        idx_r = self_right[idx]
        diff_l = self_diff[idx_r]

        if diff_l == 1:  # 複回転
            idx_rl = self_left[idx_r]
            diff_rl = self_diff[idx_rl]
            if diff_rl == 0:
                self_diff[idx] = 0
                self_diff[idx_r] = 0
            elif diff_rl == -1:
                self_diff[idx] = 1
                self_diff[idx_r] = 0
                self_diff[idx_rl] = 0
            else:  # diff_lr == 1
                self_diff[idx] = 0
                self_diff[idx_r] = -1
                self_diff[idx_rl] = 0

            # 部分木の大きさの計算
            self_size_l[idx_r] -= self_size_l[idx_rl] + 1
            self_size_l[idx_rl] += self_size_l[idx] + 1

            # 回転
            self_left[idx_r] = self_right[idx_rl]
            self_right[idx] = self_left[idx_rl]
            self_right[idx_rl] = idx_r
            self_left[idx_rl] = idx
            lr_container[idx_par] = idx_rl

            return 0

        else:  # 単回転
            if diff_l == 0:
                self_diff[idx] = -1
                nb = self_diff[idx_r] = 1
            else:  # diff_l == 1
                self_diff[idx] = 0
                nb = self_diff[idx_r] = 0

            # 部分木の大きさの計算
            self_size_l[idx_r] += self_size_l[idx] + 1

            # 回転
            self_right[idx] = self_left[idx_r]
            self_left[idx_r] = idx
            lr_container[idx_par] = idx_r

            return nb  # 新しい根の diff を返す

    def add(self, x):  # insert
        # x を加える
        # x が既に入ってる場合は False を、
        # そうでなければ True を返す

        idx = 0
        path = []
        path_left = []

        self_values = self.values
        self_left = self.left
        self_right = self.right

        while idx != -1:
            path.append(idx)
            value = self_values[idx]
            if x < value:
                path_left.append(idx)  # 重複を許さないので処理を後にする必要がある
                idx = self_left[idx]
            elif value < x:
                idx = self_right[idx]
            else:  # x == value
                return False  # 重複を許さない

        self.idx_new_val += 1
        self_diff = self.diff
        self_size_l = self.size_l

        idx = path[-1]
        if x < value:
            self_left[idx] = self.idx_new_val
        else:
            self_right[idx] = self.idx_new_val

        self_values.append(x)

        for idx_ in path_left:
            self_size_l[idx_] += 1

        self_diff[idx] += 1 if x < value else -1
        for idx_par in path[-2::-1]:
            diff = self_diff[idx]
            if diff == 0:
                return True
            elif diff == 2:  # 右回転
                self.rotate_right(idx_par, self_right[idx_par] == idx)
                return True
            elif diff == -2:  # 左回転
                self.rotate_left(idx_par, self_right[idx_par] == idx)
                return True
            else:
                self_diff[idx_par] += 1 if self_left[idx_par] == idx else -1
            idx = idx_par
        return True

    def remove(self, x):  # erase
        # x を削除する
        # x の存在が保証されている必要がある

        idx = 0
        path = []
        idx_x = -1

        self_values = self.values
        self_left = self.left
        self_right = self.right
        self_diff = self.diff
        self_size_l = self.size_l

        while idx != -1:
            path.append(idx)
            value = self_values[idx]
            if value < x:
                idx = self_right[idx]
            elif x < value:
                self_size_l[idx] -= 1  # 値の存在を保証しているので
                idx = self_left[idx]
            else:  # x == value
                idx_x = idx
                self_size_l[idx] -= 1
                idx = self_left[idx]

        idx_last_par, idx_last = path[-2:]

        if idx_last == idx_x:  # x に左の子が存在しない
            # 親の idx を付け替える
            if self_left[idx_last_par] == idx_x:
                self_left[idx_last_par] = self_right[idx_x]
                self_diff[idx_last_par] -= 1
            else:
                self_right[idx_last_par] = self_right[idx_x]
                self_diff[idx_last_par] += 1
        else:  # x に左の子が存在する
            # 自身の value を付け替える
            self_values[idx_x] = self_values[idx_last]
            if idx_last_par == idx_x:  # x 左 idx_last (左 _)?
                self_left[idx_last_par] = self_left[idx_last]
                self_diff[idx_last_par] -= 1
            else:  # x 左 _ 右 ... 右 idx_last (左 _)?
                self_right[idx_last_par] = self_left[idx_last]
                self_diff[idx_last_par] += 1

        self_rotate_left = self.rotate_left
        self_rotate_right = self.rotate_right
        diff = self_diff[idx_last_par]
        idx = idx_last_par
        for idx_par in path[-3::-1]:
            # assert diff == self_diff[idx]
            lr = self_right[idx_par] == idx
            if diff == 0:
                pass
            elif diff == 2:  # 右回転
                diff_ = self_rotate_right(idx_par, lr)
                if diff_ != 0:
                    return True
            elif diff == -2:  # 左回転
                diff_ = self_rotate_left(idx_par, lr)
                if diff_ != 0:
                    return True
            else:
                return True
            diff = self_diff[idx_par] = self_diff[idx_par] + (1 if lr else -1)
            idx = idx_par
        return True

    def pop(self, idx_):
        # 小さい方から idx_ 番目の要素を削除してその要素を返す（0-indexed）
        # idx_ 番目の値の存在が保証されている必要がある

        path = [0]
        idx_x = -1

        self_values = self.values
        self_left = self.left
        self_right = self.right
        self_diff = self.diff
        self_size_l = self.size_l

        sum_left = 0
        idx = self_right[0]
        while idx != -1:
            path.append(idx)
            c = sum_left + self_size_l[idx]
            if idx_ < c:
                self_size_l[idx] -= 1  # 値の存在が保証されているので
                idx = self_left[idx]
            elif c < idx_:
                idx = self_right[idx]
                sum_left = c + 1
            else:
                idx_x = idx
                x = self_values[idx]
                self_size_l[idx] -= 1  # なんで？
                idx = self_left[idx]

        idx_last_par, idx_last = path[-2:]

        if idx_last == idx_x:  # x に左の子が存在しない
            # 親の idx を付け替える
            if self_left[idx_last_par] == idx_x:
                self_left[idx_last_par] = self_right[idx_x]
                self_diff[idx_last_par] -= 1
            else:
                self_right[idx_last_par] = self_right[idx_x]
                self_diff[idx_last_par] += 1
        else:  # x に左の子が存在する
            # 自身の value を付け替える
            self_values[idx_x] = self_values[idx_last]
            if idx_last_par == idx_x:  # x 左 idx_last (左 _)?
                self_left[idx_last_par] = self_left[idx_last]
                self_diff[idx_last_par] -= 1
            else:  # x 左 _ 右 ... 右 idx_last (左 _)?
                self_right[idx_last_par] = self_left[idx_last]
                self_diff[idx_last_par] += 1

        self_rotate_left = self.rotate_left
        self_rotate_right = self.rotate_right
        diff = self_diff[idx_last_par]
        idx = idx_last_par
        for idx_par in path[-3::-1]:
            # assert diff == self_diff[idx]
            lr = self_right[idx_par] == idx
            if diff == 0:
                pass
            elif diff == 2:  # 右回転
                diff_ = self_rotate_right(idx_par, lr)
                if diff_ != 0:
                    return x
            elif diff == -2:  # 左回転
                diff_ = self_rotate_left(idx_par, lr)
                if diff_ != 0:
                    return x
            else:
                return x
            diff = self_diff[idx_par] = self_diff[idx_par] + (1 if lr else -1)
            idx = idx_par
        return x

    def __getitem__(self, idx_):
        # 小さい方から idx_ 番目の要素返す

        self_left = self.left
        self_right = self.right
        self_size_l = self.size_l

        sum_left = 0
        idx = self_right[0]
        while idx != -1:
            c = sum_left + self_size_l[idx]
            if idx_ < c:
                idx = self_left[idx]
            elif c < idx_:
                idx = self_right[idx]
                sum_left = c + 1
            else:  # c == idx_
                return self.values[idx]
        raise IndexError

    def __contains__(self, x):  # count
        # 値 x があるか

        self_left = self.left
        self_right = self.right
        self_values = self.values
        self_size_l = self.size_l

        idx = self_right[0]
        res = 0
        while idx != -1:
            value = self_values[idx]
            if value < x:
                res += self_size_l[idx] + 1
                idx = self_right[idx]
            elif x < value:
                idx = self_left[idx]
            else:
                return True  # res + self_size_l[idx]
        return False

    def bisect_left(self, x):  # lower_bound
        self_left = self.left
        self_right = self.right
        self_values = self.values
        self_size_l = self.size_l

        idx = self_right[0]
        res = 0
        while idx != -1:
            value = self_values[idx]
            if value < x:
                res += self_size_l[idx] + 1
                idx = self_right[idx]
            elif x < value:
                idx = self_left[idx]
            else:  # value == x
                return res + self_size_l[idx]
        return res

    def bisect_right(self, x):  # upper_bound
        self_left = self.left
        self_right = self.right
        self_values = self.values
        self_size_l = self.size_l

        idx = self_right[0]
        res = 0
        while idx != -1:
            value = self_values[idx]
            if value < x:
                res += self_size_l[idx] + 1
                idx = self_right[idx]
            elif x < value:
                idx = self_left[idx]
            else:  # value == x:
                return res + self_size_l[idx] + 1
        return res

    def print_tree(self, idx=0, depth=0, from_="・"):
        if idx == 0:
            idx = self.right[idx]
        if idx == -1:
            return
        self.print_tree(self.left[idx], depth + 1, "┏")
        print("\t\t" * depth + from_ + " val=[" + str(self.values[idx]) +
              "] diff=[" + str(self.diff[idx]) +
              "] size_l=[" + str(self.size_l[idx]) + "]")
        self.print_tree(self.right[idx], depth + 1, "┗")


# 検証1: https://atcoder.jp/contests/cpsco2019-s1/submissions/5788902
# 検証2: https://atcoder.jp/contests/arc033/submissions/6945940
