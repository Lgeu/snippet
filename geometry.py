def norm(x1, y1, x2, y2):
    # hypot を使ったほうが良さそう
    return ((x1-x2)**2 + (y1-y2)**2)**0.5

def d(a, b, c, x, y):
    # 点と直線の距離
    return abs(a*x + b*y + c) / (a**2 + b**2)**0.5

def line_cross(x1, y1, x2, y2, x3, y3, x4, y4):
    # 線分 AB と CD の交差判定
    def f(x, y, x1, y1, x2, y2):  # 直線上にあるとき 0 になる
        return (x1-x2)*(y-y1)+(y1-y2)*(x1-x)
    b1 = f(x3, y3, x1, y1, x2, y2) * f(x4, y4, x1, y1, x2, y2) < 0  # 点 C と点 D が直線 AB の異なる側にある
    b2 = f(x1, y1, x3, y3, x4, y4) * f(x2, y2, x3, y3, x4, y4) < 0  # 点 A と点 B が直線 CD の異なる側にある
    return b1 and b2

def get_convex_hull(points):  # 複素数
    # 凸包 Monotone Chain O(nlogn)
    # 参考: https://matsu7874.hatenablog.com/entry/2018/12/17/025713
    def det(p, q):
        return (p.conjugate()*q).imag
    points.sort(key=lambda x: (x.real, x.imag))
    ch = []
    for p in points:
        while len(ch) > 1:
            v_cur = ch[-1]-ch[-2]
            v_new = p-ch[-2]
            if det(v_cur, v_new) > 0:
                break
            ch.pop()
        ch.append(p)
    t = len(ch)
    for p in points[-2::-1]:
        while len(ch) > t:
            v_cur = ch[-1]-ch[-2]
            v_new = p-ch[-2]
            if det(v_cur, v_new) > 0:
                break
            ch.pop()
        ch.append(p)
    return ch[:-1]

def minimum_enclosing_circle(points):  # 複素数
    # 最小包含円 O(N)
    # 返り値は中心の座標と半径
    # 参考: https://tubo28.me/compprog/algorithm/minball/
    # 検証: https://atcoder.jp/contests/abc151/submissions/15834319
    from random import sample
    N = len(points)
    if N == 1:
        return points[0], 0
    points = sample(points, N)
    def cross(a, b):
        return a.real * b.imag - a.imag * b.real
    def norm2(a):
        return a.real * a.real + a.imag * a.imag
    def make_circle_3(a, b, c):
        A, B, C = norm2(b-c), norm2(c-a), norm2(a-b)
        S = cross(b-a, c-a)
        p = (A*(B+C-A)*a + B*(C+A-B)*b + C*(A+B-C)*c) / (4*S*S)
        radius = abs(p-a)
        return p, radius
    def make_circle_2(a, b):
        c = (a+b) / 2
        radius = abs(a-c)
        return c, radius
    def in_circle(point, circle):
        return abs(point-circle[0]) <= circle[1]+1e-7
    p0 = points[0]
    circle = make_circle_2(p0, points[1])
    for i, p_i in enumerate(points[2:], 2):
        if not in_circle(p_i, circle):
            circle = make_circle_2(p0, p_i)
            for j, p_j in enumerate(points[1:i], 1):
                if not in_circle(p_j, circle):
                    circle = make_circle_2(p_i, p_j)
                    for p_k in points[:j]:
                        if not in_circle(p_k, circle):
                            circle = make_circle_3(p_i, p_j, p_k)
    return circle


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


