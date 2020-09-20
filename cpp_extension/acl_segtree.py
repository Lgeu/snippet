# TODO: 更新ルールの異なる複数のセグ木を作ったときに正しく動くか検証

# 注: PyPy で普通に書いた方が速い


code_segtree = r"""
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

// >>> AtCoder >>>

#ifndef ATCODER_SEGTREE_HPP
#define ATCODER_SEGTREE_HPP 1

#include <algorithm>
#ifndef ATCODER_INTERNAL_BITOP_HPP
#define ATCODER_INTERNAL_BITOP_HPP 1

#ifdef _MSC_VER
#include <intrin.h>
#endif

namespace atcoder {

namespace internal {

// @param n `0 <= n`
// @return minimum non-negative `x` s.t. `n <= 2**x`
int ceil_pow2(int n) {
    int x = 0;
    while ((1U << x) < (unsigned int)(n)) x++;
    return x;
}

// @param n `1 <= n`
// @return minimum non-negative `x` s.t. `(n & (1 << x)) != 0`
int bsf(unsigned int n) {
#ifdef _MSC_VER
    unsigned long index;
    _BitScanForward(&index, n);
    return index;
#else
    return __builtin_ctz(n);
#endif
}

}  // namespace internal

}  // namespace atcoder

#endif  // ATCODER_INTERNAL_BITOP_HPP
#include <cassert>
#include <vector>

namespace atcoder {

template <class S, S (*op)(S, S), S (*e)()> struct segtree {
  public:
    segtree() : segtree(0) {}
    segtree(int n) : segtree(std::vector<S>(n, e())) {}
    segtree(const std::vector<S>& v) : _n(int(v.size())) {
        log = internal::ceil_pow2(_n);
        size = 1 << log;
        d = std::vector<S>(2 * size, e());
        for (int i = 0; i < _n; i++) d[size + i] = v[i];
        for (int i = size - 1; i >= 1; i--) {
            update(i);
        }
    }

    void set(int p, S x) {
        assert(0 <= p && p < _n);
        p += size;
        d[p] = x;
        for (int i = 1; i <= log; i++) update(p >> i);
    }

    S get(int p) {
        assert(0 <= p && p < _n);
        return d[p + size];
    }

    S prod(int l, int r) {
        assert(0 <= l && l <= r && r <= _n);
        S sml = e(), smr = e();
        l += size;
        r += size;

        while (l < r) {
            if (l & 1) sml = op(sml, d[l++]);
            if (r & 1) smr = op(d[--r], smr);
            l >>= 1;
            r >>= 1;
        }
        return op(sml, smr);
    }

    S all_prod() { return d[1]; }

    template <bool (*f)(S)> int max_right(int l) {
        return max_right(l, [](S x) { return f(x); });
    }
    template <class F> int max_right(int l, F f) {
        assert(0 <= l && l <= _n);
        assert(f(e()));
        if (l == _n) return _n;
        l += size;
        S sm = e();
        do {
            while (l % 2 == 0) l >>= 1;
            if (!f(op(sm, d[l]))) {
                while (l < size) {
                    l = (2 * l);
                    if (f(op(sm, d[l]))) {
                        sm = op(sm, d[l]);
                        l++;
                    }
                }
                return l - size;
            }
            sm = op(sm, d[l]);
            l++;
        } while ((l & -l) != l);
        return _n;
    }

    template <bool (*f)(S)> int min_left(int r) {
        return min_left(r, [](S x) { return f(x); });
    }
    template <class F> int min_left(int r, F f) {
        assert(0 <= r && r <= _n);
        assert(f(e()));
        if (r == 0) return 0;
        r += size;
        S sm = e();
        do {
            r--;
            while (r > 1 && (r % 2)) r >>= 1;
            if (!f(op(d[r], sm))) {
                while (r < size) {
                    r = (2 * r + 1);
                    if (f(op(d[r], sm))) {
                        sm = op(d[r], sm);
                        r--;
                    }
                }
                return r + 1 - size;
            }
            sm = op(d[r], sm);
        } while ((r & -r) != r);
        return 0;
    }

  private:
    int _n, size, log;
    std::vector<S> d;

    void update(int k) { d[k] = op(d[2 * k], d[2 * k + 1]); }
};

}  // namespace atcoder

#endif  // ATCODER_SEGTREE_HPP

// <<< AtCoder <<<

using namespace std;
using namespace atcoder;
#define PARSE_ARGS(types, ...) if(!PyArg_ParseTuple(args, types, __VA_ARGS__)) return NULL

struct AutoDecrefPtr{
    PyObject* p;
    AutoDecrefPtr(PyObject* _p) : p(_p) {};
    AutoDecrefPtr(const AutoDecrefPtr& rhs) : p(rhs.p) { Py_INCREF(p); };
    ~AutoDecrefPtr(){ Py_DECREF(p); }
    AutoDecrefPtr &operator=(const AutoDecrefPtr& rhs){
        Py_DECREF(p);
        p = rhs.p;
        Py_INCREF(p);
        return *this;
    }
};

static PyObject* segtree_op_py;
static AutoDecrefPtr segtree_op(AutoDecrefPtr a, AutoDecrefPtr b){
    auto tmp = PyObject_CallFunctionObjArgs(segtree_op_py, a.p, b.p, NULL);
    return AutoDecrefPtr(tmp);
}
static PyObject* segtree_e_py;
static AutoDecrefPtr segtree_e(){
    Py_INCREF(segtree_e_py);
    return AutoDecrefPtr(segtree_e_py);
}
static PyObject* segtree_f_py;
static bool segtree_f(AutoDecrefPtr x){
    PyObject* pyfunc_res = PyObject_CallFunctionObjArgs(segtree_f_py, x.p, NULL);
    int res = PyObject_IsTrue(pyfunc_res);
    if(res == -1) PyErr_Format(PyExc_ValueError, "error in SegTree f");
    return (bool)res;
}
struct SegTree{
    PyObject_HEAD
    segtree<AutoDecrefPtr, segtree_op, segtree_e>* seg;
    PyObject* op;
    PyObject* e;
    int n;
};

extern PyTypeObject SegTreeType;

static void SegTree_dealloc(SegTree* self){
    delete self->seg;
    Py_DECREF(self->op);
    Py_DECREF(self->e);
    Py_TYPE(self)->tp_free((PyObject*)self);
}
static PyObject* SegTree_new(PyTypeObject* type, PyObject* args, PyObject* kwds){
    SegTree* self;
    self = (SegTree*)type->tp_alloc(type, 0);
    return (PyObject*)self;
}
static inline void set_op_e(SegTree* self){
    segtree_op_py = self->op;
    segtree_e_py = self->e;
}
static int SegTree_init(SegTree* self, PyObject* args){
    if(Py_SIZE(args) != 3){
        self->op = Py_None;  // 何か入れておかないとヤバいことになる
        Py_INCREF(Py_None);
        self->e = Py_None;
        Py_INCREF(Py_None);
        PyErr_Format(PyExc_TypeError, "SegTree constructor expected 3 arguments (op, e, n), got %d", Py_SIZE(args));
        return -1;
    }
    PyObject* arg;
    if(!PyArg_ParseTuple(args, "OOO", &self->op, &self->e, &arg)) return -1;
    Py_INCREF(self->op);
    Py_INCREF(self->e);
    set_op_e(self);
    if(PyLong_Check(arg)){
        int n = (int)PyLong_AsLong(arg);
        if(PyErr_Occurred()) return -1;
        if(n < 0 || n > (int)1e8) {
            PyErr_Format(PyExc_ValueError, "constraint error in SegTree constructor (got %d)", n);
            return -1;
        }
        self->seg = new segtree<AutoDecrefPtr, segtree_op, segtree_e>(n);
        self->n = n;
    }else{
        PyObject *iterator = PyObject_GetIter(arg);
        if(iterator==NULL) return -1;
        PyObject *item;
        vector<AutoDecrefPtr> vec;
        if(Py_TYPE(arg)->tp_as_sequence != NULL) vec.reserve((int)Py_SIZE(arg));
        while(item = PyIter_Next(iterator)) {
            vec.push_back(item);
        }
        Py_DECREF(iterator);
        if (PyErr_Occurred()) return -1;
        self->seg = new segtree<AutoDecrefPtr, segtree_op, segtree_e>(vec);
        self->n = (int)vec.size();
    }
    return 0;
}
static PyObject* SegTree_set(SegTree* self, PyObject* args){
    long p;
    PyObject* x;
    PARSE_ARGS("lO", &p, &x);
    if(p < 0 || p >= self->n){
        PyErr_Format(PyExc_IndexError, "SegTree set index out of range (size=%d, index=%d)", self->n, p);
        return (PyObject*)NULL;
    }
    Py_INCREF(x);
    set_op_e(self);
    self->seg->set((int)p, AutoDecrefPtr(x));
    Py_RETURN_NONE;
}
static PyObject* SegTree_get(SegTree* self, PyObject* args){
    long p;
    PARSE_ARGS("l", &p);
    if(p < 0 || p >= self->n){
        PyErr_Format(PyExc_IndexError, "SegTree get index out of range (size=%d, index=%d)", self->n, p);
        return (PyObject*)NULL;
    }
    PyObject* res = self->seg->get((int)p).p;
    return Py_BuildValue("O", res);
}
static PyObject* SegTree_prod(SegTree* self, PyObject* args){
    long l, r;
    PARSE_ARGS("ll", &l, &r);
    set_op_e(self);
    auto res = self->seg->prod((int)l, (int)r).p;
    return Py_BuildValue("O", res);
}
static PyObject* SegTree_all_prod(SegTree* self, PyObject* args){
    PyObject* res = self->seg->all_prod().p;
    return Py_BuildValue("O", res);
}
static PyObject* SegTree_max_right(SegTree* self, PyObject* args){
    long l;
    PARSE_ARGS("lO", &l, &segtree_f_py);
    if(l < 0 || l > self->n){
        PyErr_Format(PyExc_IndexError, "SegTree max_right index out of range (size=%d, l=%d)", self->n, l);
        return (PyObject*)NULL;
    }
    set_op_e(self);
    int res = self->seg->max_right<segtree_f>((int)l);
    return Py_BuildValue("l", res);
}
static PyObject* SegTree_min_left(SegTree* self, PyObject* args){
    long r;
    PARSE_ARGS("lO", &r, &segtree_f_py);
    if(r < 0 || r > self->n){
        PyErr_Format(PyExc_IndexError, "SegTree max_right index out of range (size=%d, r=%d)", self->n, r);
        return (PyObject*)NULL;
    }
    set_op_e(self);
    int res = self->seg->min_left<segtree_f>((int)r);
    return Py_BuildValue("l", res);
}
static PyObject* SegTree_to_list(SegTree* self){
    PyObject* list = PyList_New(self->n);
    for(int i=0; i<self->n; i++){
        PyObject* val = self->seg->get(i).p;
        Py_INCREF(val);
        PyList_SET_ITEM(list, i, val);
    }
    return list;
}
static PyObject* SegTree_repr(PyObject* self){
    PyObject* list = SegTree_to_list((SegTree*)self);
    PyObject* res = PyUnicode_FromFormat("SegTree(%R)", list);
    Py_ReprLeave(self);
    Py_DECREF(list);
    return res;
}

static PyMethodDef SegTree_methods[] = {
    {"set", (PyCFunction)SegTree_set, METH_VARARGS, "Set item"},
    {"get", (PyCFunction)SegTree_get, METH_VARARGS, "Get item"},
    {"prod", (PyCFunction)SegTree_prod, METH_VARARGS, "Get item"},
    {"all_prod", (PyCFunction)SegTree_all_prod, METH_VARARGS, "Get item"},
    {"max_right", (PyCFunction)SegTree_max_right, METH_VARARGS, "Binary search on segtree"},
    {"min_left", (PyCFunction)SegTree_min_left, METH_VARARGS, "Binary search on segtree"},
    {"to_list", (PyCFunction)SegTree_to_list, METH_VARARGS, "Convert to list"},
    {NULL}  /* Sentinel */
};
PyTypeObject SegTreeType = {
    PyObject_HEAD_INIT(NULL)
    "atcoder.SegTree",                  /*tp_name*/
    sizeof(SegTree),                    /*tp_basicsize*/
    0,                                  /*tp_itemsize*/
    (destructor)SegTree_dealloc,        /*tp_dealloc*/
    0,                                  /*tp_print*/
    0,                                  /*tp_getattr*/
    0,                                  /*tp_setattr*/
    0,                                  /*reserved*/
    SegTree_repr,                       /*tp_repr*/
    0,                                  /*tp_as_number*/
    0,                                  /*tp_as_sequence*/
    0,                                  /*tp_as_mapping*/
    0,                                  /*tp_hash*/
    0,                                  /*tp_call*/
    0,                                  /*tp_str*/
    0,                                  /*tp_getattro*/
    0,                                  /*tp_setattro*/
    0,                                  /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
    0,                                  /*tp_doc*/
    0,                                  /*tp_traverse*/
    0,                                  /*tp_clear*/
    0,                                  /*tp_richcompare*/
    0,                                  /*tp_weaklistoffset*/
    0,                                  /*tp_iter*/
    0,                                  /*tp_iternext*/
    SegTree_methods,                    /*tp_methods*/
    0,                                  /*tp_members*/
    0,                                  /*tp_getset*/
    0,                                  /*tp_base*/
    0,                                  /*tp_dict*/
    0,                                  /*tp_descr_get*/
    0,                                  /*tp_descr_set*/
    0,                                  /*tp_dictoffset*/
    (initproc)SegTree_init,             /*tp_init*/
    0,                                  /*tp_alloc*/
    SegTree_new,                        /*tp_new*/
    0,                                  /*tp_free*/
    0,                                  /*tp_is_gc*/
    0,                                  /*tp_bases*/
    0,                                  /*tp_mro*/
    0,                                  /*tp_cache*/
    0,                                  /*tp_subclasses*/
    0,                                  /*tp_weaklist*/
    0,                                  /*tp_del*/
    0,                                  /*tp_version_tag*/
    0,                                  /*tp_finalize*/
};

static PyModuleDef atcodermodule = {
    PyModuleDef_HEAD_INIT,
    "atcoder",
    NULL,
    -1,
};

PyMODINIT_FUNC PyInit_atcoder(void)
{
    PyObject* m;
    if(PyType_Ready(&SegTreeType) < 0) return NULL;

    m = PyModule_Create(&atcodermodule);
    if(m == NULL) return NULL;

    Py_INCREF(&SegTreeType);
    if (PyModule_AddObject(m, "SegTree", (PyObject*)&SegTreeType) < 0) {
        Py_DECREF(&SegTreeType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
"""
code_setup = r"""
from distutils.core import setup, Extension
module = Extension(
    "atcoder",
    sources=["atcoder_library_wrapper.cpp"],
    extra_compile_args=["-O3", "-march=native", "-std=c++14"]
)
setup(
    name="atcoder-library",
    version="0.0.1",
    description="wrapper for atcoder library",
    ext_modules=[module]
)
"""

import os
import sys
if sys.argv[-1] == "ONLINE_JUDGE" or os.getcwd() != "/imojudge/sandbox":
    with open("atcoder_library_wrapper.cpp", "w") as f:
        f.write(code_segtree)
    with open("setup.py", "w") as f:
        f.write(code_setup)
    os.system(f"{sys.executable} setup.py build_ext --inplace")

from atcoder import SegTree

