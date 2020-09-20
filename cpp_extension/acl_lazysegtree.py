# TODO: メモリリーク確認
# TODO: max_right とかが正しく動くか検証
# TODO: 更新ルールの異なる複数のセグ木を作ったときに正しく動くか検証
 
 
code_lazy_segtree = r"""
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"
 
//#define ALLOW_MEMORY_LEAK  // メモリリーク許容して高速化
#define ILLEGAL_FUNC_CALL  // 違法な内部 API 呼び出しで高速化
 
// >>> AtCoder >>>
 
#ifndef ATCODER_LAZYSEGTREE_HPP
#define ATCODER_LAZYSEGTREE_HPP 1
 
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
#include <iostream>
#include <vector>
namespace atcoder {
 
template <class S,
          S (*op)(S, S),
          S (*e)(),
          class F,
          S (*mapping)(F, S),
          F (*composition)(F, F),
          F (*id)()>
struct lazy_segtree {
  public:
    lazy_segtree() : lazy_segtree(0) {}
    lazy_segtree(int n) : lazy_segtree(std::vector<S>(n, e())) {}
    lazy_segtree(const std::vector<S>& v) : _n(int(v.size())) {
        log = internal::ceil_pow2(_n);
        size = 1 << log;
        d = std::vector<S>(2 * size, e());
        lz = std::vector<F>(size, id());
        for (int i = 0; i < _n; i++) d[size + i] = v[i];
        for (int i = size - 1; i >= 1; i--) {
            update(i);
        }
    }
 
    void set(int p, S x) {
        assert(0 <= p && p < _n);
        p += size;
        for (int i = log; i >= 1; i--) push(p >> i);
        d[p] = x;
        for (int i = 1; i <= log; i++) update(p >> i);
    }
 
    S get(int p) {
        assert(0 <= p && p < _n);
        p += size;
        for (int i = log; i >= 1; i--) push(p >> i);
        return d[p];
    }
 
    S prod(int l, int r) {
        assert(0 <= l && l <= r && r <= _n);
        if (l == r) return e();
 
        l += size;
        r += size;
 
        for (int i = log; i >= 1; i--) {
            if (((l >> i) << i) != l) push(l >> i);
            if (((r >> i) << i) != r) push(r >> i);
        }
 
        S sml = e(), smr = e();
        while (l < r) {
            if (l & 1) sml = op(sml, d[l++]);
            if (r & 1) smr = op(d[--r], smr);
            l >>= 1;
            r >>= 1;
        }
 
        return op(sml, smr);
    }
 
    S all_prod() { return d[1]; }
 
    void apply(int p, F f) {
        assert(0 <= p && p < _n);
        p += size;
        for (int i = log; i >= 1; i--) push(p >> i);
        d[p] = mapping(f, d[p]);
        for (int i = 1; i <= log; i++) update(p >> i);
    }
    void apply(int l, int r, F f) {
        assert(0 <= l && l <= r && r <= _n);
        if (l == r) return;
 
        l += size;
        r += size;
 
        for (int i = log; i >= 1; i--) {
            if (((l >> i) << i) != l) push(l >> i);
            if (((r >> i) << i) != r) push((r - 1) >> i);
        }
 
        {
            int l2 = l, r2 = r;
            while (l < r) {
                if (l & 1) all_apply(l++, f);
                if (r & 1) all_apply(--r, f);
                l >>= 1;
                r >>= 1;
            }
            l = l2;
            r = r2;
        }
 
        for (int i = 1; i <= log; i++) {
            if (((l >> i) << i) != l) update(l >> i);
            if (((r >> i) << i) != r) update((r - 1) >> i);
        }
    }
 
    template <bool (*g)(S)> int max_right(int l) {
        return max_right(l, [](S x) { return g(x); });
    }
    template <class G> int max_right(int l, G g) {
        assert(0 <= l && l <= _n);
        assert(g(e()));
        if (l == _n) return _n;
        l += size;
        for (int i = log; i >= 1; i--) push(l >> i);
        S sm = e();
        do {
            while (l % 2 == 0) l >>= 1;
            if (!g(op(sm, d[l]))) {
                while (l < size) {
                    push(l);
                    l = (2 * l);
                    if (g(op(sm, d[l]))) {
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
 
    template <bool (*g)(S)> int min_left(int r) {
        return min_left(r, [](S x) { return g(x); });
    }
    template <class G> int min_left(int r, G g) {
        assert(0 <= r && r <= _n);
        assert(g(e()));
        if (r == 0) return 0;
        r += size;
        for (int i = log; i >= 1; i--) push((r - 1) >> i);
        S sm = e();
        do {
            r--;
            while (r > 1 && (r % 2)) r >>= 1;
            if (!g(op(d[r], sm))) {
                while (r < size) {
                    push(r);
                    r = (2 * r + 1);
                    if (g(op(d[r], sm))) {
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
    std::vector<F> lz;
 
    void update(int k) { d[k] = op(d[2 * k], d[2 * k + 1]); }
    void all_apply(int k, F f) {
        d[k] = mapping(f, d[k]);
        if (k < size) lz[k] = composition(f, lz[k]);
    }
    void push(int k) {
        all_apply(2 * k, lz[k]);
        all_apply(2 * k + 1, lz[k]);
        lz[k] = id();
    }
};
 
}  // namespace atcoder
 
#endif  // ATCODER_LAZYSEGTREE_HPP
 
// <<< AtCoder <<<
 
using namespace std;
using namespace atcoder;
#define PARSE_ARGS(types, ...) if(!PyArg_ParseTuple(args, types, __VA_ARGS__)) return NULL
 
struct AutoDecrefPtr{
    PyObject* p;
    AutoDecrefPtr(PyObject* _p) : p(_p) {};
#ifndef ALLOW_MEMORY_LEAK
    AutoDecrefPtr(const AutoDecrefPtr& rhs) : p(rhs.p) { Py_INCREF(p); };
    ~AutoDecrefPtr(){ Py_DECREF(p); }
    AutoDecrefPtr &operator=(const AutoDecrefPtr& rhs){
        Py_DECREF(p);
        p = rhs.p;
        Py_INCREF(p);
        return *this;
    }
#endif
};
 
#ifdef ILLEGAL_FUNC_CALL
static PyObject* py_function_args[2];
#endif
 
// >>> functions for laze_segtree constructor >>>
static PyObject* lazy_segtree_op_py;
static AutoDecrefPtr lazy_segtree_op(AutoDecrefPtr a, AutoDecrefPtr b){
#ifdef ILLEGAL_FUNC_CALL
    py_function_args[0] = a.p;
    py_function_args[1] = b.p;
    PyObject* res = _PyObject_FastCall(lazy_segtree_op_py, py_function_args, 2);
#else
    PyObject* res(PyObject_CallFunctionObjArgs(lazy_segtree_op_py, a.p, b.p, NULL));
#endif
    Py_INCREF(res);  // ???????????????
    return AutoDecrefPtr(res);
}
static PyObject* lazy_segtree_e_py;
static AutoDecrefPtr lazy_segtree_e(){
    Py_INCREF(lazy_segtree_e_py);
    return AutoDecrefPtr(lazy_segtree_e_py);
}
static PyObject* lazy_segtree_mapping_py;
static AutoDecrefPtr lazy_segtree_mapping(AutoDecrefPtr f, AutoDecrefPtr x){
#ifdef ILLEGAL_FUNC_CALL
    py_function_args[0] = f.p;
    py_function_args[1] = x.p;
    PyObject* res = _PyObject_FastCall(lazy_segtree_mapping_py, py_function_args, 2);
    return AutoDecrefPtr(res);
#else
    return AutoDecrefPtr(PyObject_CallFunctionObjArgs(lazy_segtree_mapping_py, f.p, x.p, NULL));
#endif
}
static PyObject* lazy_segtree_composition_py;
static AutoDecrefPtr lazy_segtree_composition(AutoDecrefPtr f, AutoDecrefPtr g){
#ifdef ILLEGAL_FUNC_CALL
    py_function_args[0] = f.p;
    py_function_args[1] = g.p;
    PyObject* res = _PyObject_FastCall(lazy_segtree_composition_py, py_function_args, 2);
    return AutoDecrefPtr(res);
#else
    return AutoDecrefPtr(PyObject_CallFunctionObjArgs(lazy_segtree_composition_py, f.p, g.p, NULL));
#endif
}
static PyObject* lazy_segtree_id_py;
static AutoDecrefPtr lazy_segtree_id(){
    Py_INCREF(lazy_segtree_id_py);
    return AutoDecrefPtr(lazy_segtree_id_py);
}
using lazyseg = lazy_segtree<AutoDecrefPtr,
                             lazy_segtree_op,
                             lazy_segtree_e,
                             AutoDecrefPtr,
                             lazy_segtree_mapping,
                             lazy_segtree_composition,
                             lazy_segtree_id>;
// <<< functions for laze_segtree constructor <<<
 
static PyObject* lazy_segtree_f_py;
static bool lazy_segtree_f(AutoDecrefPtr x){
    PyObject* pyfunc_res = PyObject_CallFunctionObjArgs(lazy_segtree_f_py, x.p, NULL);
    int res = PyObject_IsTrue(pyfunc_res);
    if(res == -1) PyErr_Format(PyExc_ValueError, "error in LazySegTree f");
    return (bool)res;
}
 
struct LazySegTree{
    PyObject_HEAD
    lazyseg* seg;
    PyObject* op;
    PyObject* e;
    PyObject* mapping;
    PyObject* composition;
    PyObject* id;
    int n;
};
static inline void set_rules(LazySegTree* self){
    lazy_segtree_op_py = self->op;
    lazy_segtree_e_py = self->e;
    lazy_segtree_mapping_py = self->mapping;
    lazy_segtree_composition_py = self->composition;
    lazy_segtree_id_py = self->id;
}
 
// >>> LazySegTree functions >>>
 
extern PyTypeObject LazySegTreeType;
 
static void LazySegTree_dealloc(LazySegTree* self){
    delete self->seg;
    Py_DECREF(self->op);
    Py_DECREF(self->e);
    Py_DECREF(self->mapping);
    Py_DECREF(self->composition);
    Py_DECREF(self->id);
    Py_TYPE(self)->tp_free((PyObject*)self);
}
static PyObject* LazySegTree_new(PyTypeObject* type, PyObject* args, PyObject* kwds){
    LazySegTree* self;
    self = (LazySegTree*)type->tp_alloc(type, 0);
    return (PyObject*)self;
}
static int LazySegTree_init(LazySegTree* self, PyObject* args){
    if(Py_SIZE(args) != 6){
        self->op = Py_None;  // 何か入れておかないとヤバいことになる
        Py_INCREF(Py_None);
        self->e = Py_None;
        Py_INCREF(Py_None);
        self->mapping = Py_None;
        Py_INCREF(Py_None);
        self->composition = Py_None;
        Py_INCREF(Py_None);
        self->id = Py_None;
        Py_INCREF(Py_None);
        PyErr_Format(PyExc_TypeError,
                     "LazySegTree constructor expected 6 arguments (op, e, mapping, composition, identity, n), got %d", Py_SIZE(args));
        return -1;
    }
    PyObject* arg;
    if(!PyArg_ParseTuple(args, "OOOOOO", 
                         &self->op, &self->e,
                         &self->mapping, &self->composition, &self->id, &arg)) return -1;
    Py_INCREF(self->op);
    Py_INCREF(self->e);
    Py_INCREF(self->mapping);
    Py_INCREF(self->composition);
    Py_INCREF(self->id);
    set_rules(self);
    if(PyLong_Check(arg)){
        int n = (int)PyLong_AsLong(arg);
        if(PyErr_Occurred()) return -1;
        if(n < 0 || n > (int)1e8) {
            PyErr_Format(PyExc_ValueError, "constraint error in LazySegTree constructor (got %d)", n);
            return -1;
        }
        self->seg = new lazyseg(n);
        self->n = n;
    }else{
        PyObject *iterator = PyObject_GetIter(arg);
        if(iterator==NULL) return -1;
        PyObject *item;
        vector<AutoDecrefPtr> vec;
        if(Py_TYPE(arg)->tp_as_sequence != NULL) vec.reserve((int)Py_SIZE(arg));
        while(item = PyIter_Next(iterator)) {
            vec.emplace_back(item);
        }
        Py_DECREF(iterator);
        if (PyErr_Occurred()) return -1;
        self->seg = new lazyseg(vec);
        self->n = (int)vec.size();
    }
    return 0;
}
static PyObject* LazySegTree_set(LazySegTree* self, PyObject* args){
    long p;
    PyObject* x;
    PARSE_ARGS("lO", &p, &x);
    if(p < 0 || p >= self->n){
        PyErr_Format(PyExc_IndexError, "LazySegTree set index out of range (size=%d, index=%d)", self->n, p);
        return (PyObject*)NULL;
    }
    Py_INCREF(x);
    set_rules(self);
    self->seg->set((int)p, AutoDecrefPtr(x));
    Py_RETURN_NONE;
}
static PyObject* LazySegTree_get(LazySegTree* self, PyObject* args){
    long p;
    PARSE_ARGS("l", &p);
    if(p < 0 || p >= self->n){
        PyErr_Format(PyExc_IndexError, "LazySegTree get index out of range (size=%d, index=%d)", self->n, p);
        return (PyObject*)NULL;
    }
    set_rules(self);
    PyObject* res = self->seg->get((int)p).p;
    return Py_BuildValue("O", res);
}
static PyObject* LazySegTree_prod(LazySegTree* self, PyObject* args){
    long l, r;
    PARSE_ARGS("ll", &l, &r);
    set_rules(self);
    PyObject* res = self->seg->prod((int)l, (int)r).p;
    return Py_BuildValue("O", res);
}
static PyObject* LazySegTree_prod_getitem(LazySegTree* self, PyObject* args){
    long l, r, idx;
    PARSE_ARGS("lll", &l, &r, &idx);
    set_rules(self);
    PyObject* res = self->seg->prod((int)l, (int)r).p;
    res = PySequence_Fast_GET_ITEM(res, idx);  // 要素がタプルと仮定
    return Py_BuildValue("O", res);
}
static PyObject* LazySegTree_all_prod(LazySegTree* self, PyObject* args){
    PyObject* res = self->seg->all_prod().p;
    return Py_BuildValue("O", res);
}
static PyObject* LazySegTree_apply(LazySegTree* self, PyObject* args){
    if(Py_SIZE(args) == 3){
        long l, r;
        PyObject* x;
        PARSE_ARGS("llO", &l, &r, &x);
        Py_INCREF(x);
        set_rules(self);
        self->seg->apply(l, r, AutoDecrefPtr(x));
        Py_RETURN_NONE;
    }else if(Py_SIZE(args) == 2){
        long p;
        PyObject* x;
        PARSE_ARGS("lO", &p, &x);
        if(p < 0 || p >= self->n){
            PyErr_Format(PyExc_IndexError, "LazySegTree apply index out of range (size=%d, index=%d)", self->n, p);
            return (PyObject*)NULL;
        }
        Py_INCREF(x);
        set_rules(self);
        self->seg->apply(p, AutoDecrefPtr(x));
        Py_RETURN_NONE;
    }else{
        PyErr_Format(PyExc_TypeError,
            "LazySegTree apply expected 2 (p, x) or 3 (l, r, x) arguments, got %d", Py_SIZE(args));
        return (PyObject*)NULL;
    }
}
static PyObject* LazySegTree_max_right(LazySegTree* self, PyObject* args){
    long l;
    PARSE_ARGS("lO", &l, &lazy_segtree_f_py);
    if(l < 0 || l > self->n){
        PyErr_Format(PyExc_IndexError, "LazySegTree max_right index out of range (size=%d, l=%d)", self->n, l);
        return (PyObject*)NULL;
    }
    set_rules(self);
    int res = self->seg->max_right<lazy_segtree_f>((int)l);
    return Py_BuildValue("l", res);
}
static PyObject* LazySegTree_min_left(LazySegTree* self, PyObject* args){
    long r;
    PARSE_ARGS("lO", &r, &lazy_segtree_f_py);
    if(r < 0 || r > self->n){
        PyErr_Format(PyExc_IndexError, "LazySegTree max_right index out of range (size=%d, r=%d)", self->n, r);
        return (PyObject*)NULL;
    }
    set_rules(self);
    int res = self->seg->min_left<lazy_segtree_f>((int)r);
    return Py_BuildValue("l", res);
}
static PyObject* LazySegTree_to_list(LazySegTree* self){
    PyObject* list = PyList_New(self->n);
    for(int i=0; i<self->n; i++){
        PyObject* val = self->seg->get(i).p;
        Py_INCREF(val);
        PyList_SET_ITEM(list, i, val);
    }
    return list;
}
static PyObject* LazySegTree_repr(PyObject* self){
    PyObject* list = LazySegTree_to_list((LazySegTree*)self);
    PyObject* res = PyUnicode_FromFormat("LazySegTree(%R)", list);
    Py_ReprLeave(self);
    Py_DECREF(list);
    return res;
}
// <<< LazySegTree functions <<<
 
static PyMethodDef LazySegTree_methods[] = {
    {"set", (PyCFunction)LazySegTree_set, METH_VARARGS, "Set item"},
    {"get", (PyCFunction)LazySegTree_get, METH_VARARGS, "Get item"},
    {"prod", (PyCFunction)LazySegTree_prod, METH_VARARGS, "Get item"},
    {"prod_getitem", (PyCFunction)LazySegTree_prod_getitem, METH_VARARGS, "Get item"},
    {"all_prod", (PyCFunction)LazySegTree_all_prod, METH_VARARGS, "Get item"},
    {"apply", (PyCFunction)LazySegTree_apply, METH_VARARGS, "Apply function"},
    {"max_right", (PyCFunction)LazySegTree_max_right, METH_VARARGS, "Binary search on lazy segtree"},
    {"min_left", (PyCFunction)LazySegTree_min_left, METH_VARARGS, "Binary search on lazy segtree"},
    {"to_list", (PyCFunction)LazySegTree_to_list, METH_VARARGS, "Convert to list"},
    {NULL}  /* Sentinel */
};
PyTypeObject LazySegTreeType = {
    PyObject_HEAD_INIT(NULL)
    "atcoder.LazySegTree",              /*tp_name*/
    sizeof(LazySegTree),                /*tp_basicsize*/
    0,                                  /*tp_itemsize*/
    (destructor)LazySegTree_dealloc,    /*tp_dealloc*/
    0,                                  /*tp_print*/
    0,                                  /*tp_getattr*/
    0,                                  /*tp_setattr*/
    0,                                  /*reserved*/
    LazySegTree_repr,                   /*tp_repr*/
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
    LazySegTree_methods,                /*tp_methods*/
    0,                                  /*tp_members*/
    0,                                  /*tp_getset*/
    0,                                  /*tp_base*/
    0,                                  /*tp_dict*/
    0,                                  /*tp_descr_get*/
    0,                                  /*tp_descr_set*/
    0,                                  /*tp_dictoffset*/
    (initproc)LazySegTree_init,         /*tp_init*/
    0,                                  /*tp_alloc*/
    LazySegTree_new,                    /*tp_new*/
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
    if(PyType_Ready(&LazySegTreeType) < 0) return NULL;
 
    m = PyModule_Create(&atcodermodule);
    if(m == NULL) return NULL;
 
    Py_INCREF(&LazySegTreeType);
    if (PyModule_AddObject(m, "LazySegTree", (PyObject*)&LazySegTreeType) < 0) {
        Py_DECREF(&LazySegTreeType);
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
        f.write(code_lazy_segtree)
    with open("setup.py", "w") as f:
        f.write(code_setup)
    os.system(f"{sys.executable} setup.py build_ext --inplace")
 
from atcoder import LazySegTree
