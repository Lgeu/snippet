# TODO: メモリリーク確認

code_acl_math = r"""
#define PY_SSIZE_T_CLEAN
#include <Python.h>


// >>> AtCoder >>>

#ifndef ATCODER_MATH_HPP
#define ATCODER_MATH_HPP 1

#include <algorithm>
#include <cassert>
#include <tuple>
#include <vector>

#ifndef ATCODER_INTERNAL_MATH_HPP
#define ATCODER_INTERNAL_MATH_HPP 1

#include <utility>

#ifdef _MSC_VER
#include <intrin.h>
#endif

namespace atcoder {

namespace internal {

// @param m `1 <= m`
// @return x mod m
constexpr long long safe_mod(long long x, long long m) {
    x %= m;
    if (x < 0) x += m;
    return x;
}

// Fast modular multiplication by barrett reduction
// Reference: https://en.wikipedia.org/wiki/Barrett_reduction
// NOTE: reconsider after Ice Lake
struct barrett {
    unsigned int _m;
    unsigned long long im;

    // @param m `1 <= m < 2^31`
    barrett(unsigned int m) : _m(m), im((unsigned long long)(-1) / m + 1) {}

    // @return m
    unsigned int umod() const { return _m; }

    // @param a `0 <= a < m`
    // @param b `0 <= b < m`
    // @return `a * b % m`
    unsigned int mul(unsigned int a, unsigned int b) const {
        // [1] m = 1
        // a = b = im = 0, so okay

        // [2] m >= 2
        // im = ceil(2^64 / m)
        // -> im * m = 2^64 + r (0 <= r < m)
        // let z = a*b = c*m + d (0 <= c, d < m)
        // a*b * im = (c*m + d) * im = c*(im*m) + d*im = c*2^64 + c*r + d*im
        // c*r + d*im < m * m + m * im < m * m + 2^64 + m <= 2^64 + m * (m + 1) < 2^64 * 2
        // ((ab * im) >> 64) == c or c + 1
        unsigned long long z = a;
        z *= b;
#ifdef _MSC_VER
        unsigned long long x;
        _umul128(z, im, &x);
#else
        unsigned long long x =
            (unsigned long long)(((unsigned __int128)(z)*im) >> 64);
#endif
        unsigned int v = (unsigned int)(z - x * _m);
        if (_m <= v) v += _m;
        return v;
    }
};

// @param n `0 <= n`
// @param m `1 <= m`
// @return `(x ** n) % m`
constexpr long long pow_mod_constexpr(long long x, long long n, int m) {
    if (m == 1) return 0;
    unsigned int _m = (unsigned int)(m);
    unsigned long long r = 1;
    unsigned long long y = safe_mod(x, m);
    while (n) {
        if (n & 1) r = (r * y) % _m;
        y = (y * y) % _m;
        n >>= 1;
    }
    return r;
}

// Reference:
// M. Forisek and J. Jancina,
// Fast Primality Testing for Integers That Fit into a Machine Word
// @param n `0 <= n`
constexpr bool is_prime_constexpr(int n) {
    if (n <= 1) return false;
    if (n == 2 || n == 7 || n == 61) return true;
    if (n % 2 == 0) return false;
    long long d = n - 1;
    while (d % 2 == 0) d /= 2;
    constexpr long long bases[3] = {2, 7, 61};
    for (long long a : bases) {
        long long t = d;
        long long y = pow_mod_constexpr(a, t, n);
        while (t != n - 1 && y != 1 && y != n - 1) {
            y = y * y % n;
            t <<= 1;
        }
        if (y != n - 1 && t % 2 == 0) {
            return false;
        }
    }
    return true;
}
template <int n> constexpr bool is_prime = is_prime_constexpr(n);

// @param b `1 <= b`
// @return pair(g, x) s.t. g = gcd(a, b), xa = g (mod b), 0 <= x < b/g
constexpr std::pair<long long, long long> inv_gcd(long long a, long long b) {
    a = safe_mod(a, b);
    if (a == 0) return {b, 0};

    // Contracts:
    // [1] s - m0 * a = 0 (mod b)
    // [2] t - m1 * a = 0 (mod b)
    // [3] s * |m1| + t * |m0| <= b
    long long s = b, t = a;
    long long m0 = 0, m1 = 1;

    while (t) {
        long long u = s / t;
        s -= t * u;
        m0 -= m1 * u;  // |m1 * u| <= |m1| * s <= b

        // [3]:
        // (s - t * u) * |m1| + t * |m0 - m1 * u|
        // <= s * |m1| - t * u * |m1| + t * (|m0| + |m1| * u)
        // = s * |m1| + t * |m0| <= b

        auto tmp = s;
        s = t;
        t = tmp;
        tmp = m0;
        m0 = m1;
        m1 = tmp;
    }
    // by [3]: |m0| <= b/g
    // by g != b: |m0| < b/g
    if (m0 < 0) m0 += b / s;
    return {s, m0};
}

// Compile time primitive root
// @param m must be prime
// @return primitive root (and minimum in now)
constexpr int primitive_root_constexpr(int m) {
    if (m == 2) return 1;
    if (m == 167772161) return 3;
    if (m == 469762049) return 3;
    if (m == 754974721) return 11;
    if (m == 998244353) return 3;
    int divs[20] = {};
    divs[0] = 2;
    int cnt = 1;
    int x = (m - 1) / 2;
    while (x % 2 == 0) x /= 2;
    for (int i = 3; (long long)(i)*i <= x; i += 2) {
        if (x % i == 0) {
            divs[cnt++] = i;
            while (x % i == 0) {
                x /= i;
            }
        }
    }
    if (x > 1) {
        divs[cnt++] = x;
    }
    for (int g = 2;; g++) {
        bool ok = true;
        for (int i = 0; i < cnt; i++) {
            if (pow_mod_constexpr(g, (m - 1) / divs[i], m) == 1) {
                ok = false;
                break;
            }
        }
        if (ok) return g;
    }
}
template <int m> constexpr int primitive_root = primitive_root_constexpr(m);

}  // namespace internal

}  // namespace atcoder

#endif  // ATCODER_INTERNAL_MATH_HPP

namespace atcoder {

long long pow_mod(long long x, long long n, int m) {
    assert(0 <= n && 1 <= m);
    if (m == 1) return 0;
    internal::barrett bt((unsigned int)(m));
    unsigned int r = 1, y = (unsigned int)(internal::safe_mod(x, m));
    while (n) {
        if (n & 1) r = bt.mul(r, y);
        y = bt.mul(y, y);
        n >>= 1;
    }
    return r;
}

long long inv_mod(long long x, long long m) {
    assert(1 <= m);
    auto z = internal::inv_gcd(x, m);
    assert(z.first == 1);
    return z.second;
}

// (rem, mod)
std::pair<long long, long long> crt(const std::vector<long long>& r,
                                    const std::vector<long long>& m) {
    assert(r.size() == m.size());
    int n = int(r.size());
    // Contracts: 0 <= r0 < m0
    long long r0 = 0, m0 = 1;
    for (int i = 0; i < n; i++) {
        assert(1 <= m[i]);
        long long r1 = internal::safe_mod(r[i], m[i]), m1 = m[i];
        if (m0 < m1) {
            std::swap(r0, r1);
            std::swap(m0, m1);
        }
        if (m0 % m1 == 0) {
            if (r0 % m1 != r1) return {0, 0};
            continue;
        }
        // assume: m0 > m1, lcm(m0, m1) >= 2 * max(m0, m1)

        // (r0, m0), (r1, m1) -> (r2, m2 = lcm(m0, m1));
        // r2 % m0 = r0
        // r2 % m1 = r1
        // -> (r0 + x*m0) % m1 = r1
        // -> x*u0*g % (u1*g) = (r1 - r0) (u0*g = m0, u1*g = m1)
        // -> x = (r1 - r0) / g * inv(u0) (mod u1)

        // im = inv(u0) (mod u1) (0 <= im < u1)
        long long g, im;
        std::tie(g, im) = internal::inv_gcd(m0, m1);

        long long u1 = (m1 / g);
        // |r1 - r0| < (m0 + m1) <= lcm(m0, m1)
        if ((r1 - r0) % g) return {0, 0};

        // u1 * u1 <= m1 * m1 / g / g <= m0 * m1 / g = lcm(m0, m1)
        long long x = (r1 - r0) / g % u1 * im % u1;

        // |r0| + |m0 * x|
        // < m0 + m0 * (u1 - 1)
        // = m0 + m0 * m1 / g - m0
        // = lcm(m0, m1)
        r0 += x * m0;
        m0 *= u1;  // -> lcm(m0, m1)
        if (r0 < 0) r0 += m0;
    }
    return {r0, m0};
}

long long floor_sum(long long n, long long m, long long a, long long b) {
    long long ans = 0;
    if (a >= m) {
        ans += (n - 1) * n * (a / m) / 2;
        a %= m;
    }
    if (b >= m) {
        ans += n * (b / m);
        b %= m;
    }

    long long y_max = (a * n + b) / m, x_max = (y_max * m - b);
    if (y_max == 0) return ans;
    ans += (n - (x_max + a - 1) / a) * y_max;
    ans += floor_sum(y_max, a, m, (a - x_max % a) % a);
    return ans;
}

}  // namespace atcoder

#endif  // ATCODER_MATH_HPP

// <<< AtCoder <<<


using namespace std;
using namespace atcoder;
#define PARSE_ARGS(types, ...) if(!PyArg_ParseTuple(args, types, __VA_ARGS__)) return NULL


// >>> acl_math definition >>>

static PyObject* acl_math_pow_mod(PyObject* self, PyObject* args){
    long long x, n;
    long m;
    PARSE_ARGS("LLl", &x, &n, &m);
    if(n < 0 || m <= 0){
        PyErr_Format(PyExc_IndexError,
            "pow_mod constraint error (costraint: 0<=n, 1<=m, got x=%lld, n=%lld, m=%d)", x, n, m);
        return (PyObject*)NULL;
    }
    return Py_BuildValue("L", pow_mod(x, n, m));
}
static PyObject* acl_math_inv_mod(PyObject* self, PyObject* args){
    long long x, m;
    PARSE_ARGS("LL", &x, &m);
    if(m <= 0){
        PyErr_Format(PyExc_IndexError,
            "inv_mod constraint error (costraint: 1<=m, got x=%lld, m=%d)", x, m);
        return (PyObject*)NULL;
    }
    return Py_BuildValue("L", inv_mod(x, m));
}
static PyObject* acl_math_crt(PyObject* self, PyObject* args){
    PyObject *r_iterable, *m_iterable, *iterator, *item;
    PARSE_ARGS("OO", &r_iterable, &m_iterable);
    vector<long long> r, m;
    
    iterator = PyObject_GetIter(r_iterable);
    if(iterator==NULL) return NULL;
    if(Py_TYPE(r_iterable)->tp_as_sequence != NULL) r.reserve((int)Py_SIZE(r_iterable));
    while(item = PyIter_Next(iterator)) {
        const long long& ri = PyLong_AsLongLong(item);
        r.push_back(ri);
        Py_DECREF(item);
    }
    Py_DECREF(iterator);
    if (PyErr_Occurred()) return NULL;
    
    iterator = PyObject_GetIter(m_iterable);
    if(iterator==NULL) return NULL;
    if(Py_TYPE(m_iterable)->tp_as_sequence != NULL) m.reserve((int)Py_SIZE(m_iterable));
    while(item = PyIter_Next(iterator)) {
        const long long& mi = PyLong_AsLongLong(item);
        if(mi <= 0 && !PyErr_Occurred()) PyErr_Format(PyExc_ValueError,
            "crt constraint error (constraint: m>=1, got %lld)", mi);
        m.push_back(mi);
        Py_DECREF(item);
    }
    Py_DECREF(iterator);
    if (PyErr_Occurred()) return NULL;
    
    if(r.size() != m.size()){
        PyErr_Format(PyExc_ValueError,
            "crt constraint error (constraint: len(r)=len(m), got len(r)=%d, len(m)=%d)", r.size(), m.size());
        return NULL;
    }
    const pair<long long, long long>& res = crt(r, m);
    return Py_BuildValue("LL", res.first, res.second);
}
static PyObject* acl_math_floor_sum(PyObject* self, PyObject* args){
    long long n, m, a, b;
    PARSE_ARGS("LLLL", &n, &m, &a, &b);
    if(n < 0 || n > (long long)1e9 || m <= 0 || m > (long long)1e9 || a < 0 || a >= m || b < 0 || b >= m){
        PyErr_Format(PyExc_IndexError,
            "floor_sum constraint error (costraint: 0<=n<=1e9, 1<=m<=1e9, 0<=a,b<m, "
            "got n=%lld, m=%lld, a=%lld, b=%lld)", n, m, a, b);
        return (PyObject*)NULL;
    }
    return Py_BuildValue("L", floor_sum(n, m, a, b));
}

static PyMethodDef acl_math_methods[] = {
    {"pow_mod", (PyCFunction)acl_math_pow_mod, METH_VARARGS, "pow_mod"},
    {"inv_mod", (PyCFunction)acl_math_inv_mod, METH_VARARGS, "inv_mod"},
    {"crt", (PyCFunction)acl_math_crt, METH_VARARGS, "crt"},
    {"floor_sum", (PyCFunction)acl_math_floor_sum, METH_VARARGS, "floor_sum"},
    {NULL}  /* Sentinel */
};


// <<< acl_math definition <<<


static PyModuleDef acl_math_module = {
    PyModuleDef_HEAD_INIT,
    "acl_math",
    NULL,
    -1,
    acl_math_methods,
};

PyMODINIT_FUNC PyInit_acl_math(void){
    return PyModule_Create(&acl_math_module);
}
"""
code_acl_math_setup = r"""
from distutils.core import setup, Extension
module = Extension(
    "acl_math",
    sources=["acl_math.cpp"],
    extra_compile_args=["-O3", "-march=native", "-std=c++14"]
)
setup(
    name="acl_math",
    version="0.0.1",
    description="wrapper for atcoder library math",
    ext_modules=[module]
)
"""

import os
import sys
if sys.argv[-1] == "ONLINE_JUDGE" or os.getcwd() != "/imojudge/sandbox":
    with open("acl_math.cpp", "w") as f:
        f.write(code_acl_math)
    with open("acl_math_setup.py", "w") as f:
        f.write(code_acl_math_setup)
    os.system(f"{sys.executable} acl_math_setup.py build_ext --inplace")

from acl_math import pow_mod, inv_mod, crt, floor_sum
