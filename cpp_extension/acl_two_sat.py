# TODO: メモリリーク確認
# TODO: __repr__ を書く

code_two_sat = r"""
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

// 元のライブラリの private を剥がした

// >>> AtCoder >>>

#ifndef ATCODER_TWOSAT_HPP
#define ATCODER_TWOSAT_HPP 1

#ifndef ATCODER_INTERNAL_SCC_HPP
#define ATCODER_INTERNAL_SCC_HPP 1

#include <algorithm>
#include <utility>
#include <vector>

namespace atcoder {
namespace internal {

template <class E> struct csr {
    std::vector<int> start;
    std::vector<E> elist;
    csr(int n, const std::vector<std::pair<int, E>>& edges)
        : start(n + 1), elist(edges.size()) {
        for (auto e : edges) {
            start[e.first + 1]++;
        }
        for (int i = 1; i <= n; i++) {
            start[i] += start[i - 1];
        }
        auto counter = start;
        for (auto e : edges) {
            elist[counter[e.first]++] = e.second;
        }
    }
};

// Reference:
// R. Tarjan,
// Depth-First Search and Linear Graph Algorithms
struct scc_graph {
  public:
    scc_graph(int n) : _n(n) {}

    int num_vertices() { return _n; }

    void add_edge(int from, int to) { edges.push_back({from, {to}}); }

    // @return pair of (# of scc, scc id)
    std::pair<int, std::vector<int>> scc_ids() {
        auto g = csr<edge>(_n, edges);
        int now_ord = 0, group_num = 0;
        std::vector<int> visited, low(_n), ord(_n, -1), ids(_n);
        visited.reserve(_n);
        auto dfs = [&](auto self, int v) -> void {
            low[v] = ord[v] = now_ord++;
            visited.push_back(v);
            for (int i = g.start[v]; i < g.start[v + 1]; i++) {
                auto to = g.elist[i].to;
                if (ord[to] == -1) {
                    self(self, to);
                    low[v] = std::min(low[v], low[to]);
                } else {
                    low[v] = std::min(low[v], ord[to]);
                }
            }
            if (low[v] == ord[v]) {
                while (true) {
                    int u = visited.back();
                    visited.pop_back();
                    ord[u] = _n;
                    ids[u] = group_num;
                    if (u == v) break;
                }
                group_num++;
            }
        };
        for (int i = 0; i < _n; i++) {
            if (ord[i] == -1) dfs(dfs, i);
        }
        for (auto& x : ids) {
            x = group_num - 1 - x;
        }
        return {group_num, ids};
    }

    std::vector<std::vector<int>> scc() {
        auto ids = scc_ids();
        int group_num = ids.first;
        std::vector<int> counts(group_num);
        for (auto x : ids.second) counts[x]++;
        std::vector<std::vector<int>> groups(ids.first);
        for (int i = 0; i < group_num; i++) {
            groups[i].reserve(counts[i]);
        }
        for (int i = 0; i < _n; i++) {
            groups[ids.second[i]].push_back(i);
        }
        return groups;
    }

  private:
    int _n;
    struct edge {
        int to;
    };
    std::vector<std::pair<int, edge>> edges;
};

}  // namespace internal

}  // namespace atcoder

#endif  // ATCODER_INTERNAL_SCC_HPP

#include <cassert>
#include <vector>

namespace atcoder {

// Reference:
// B. Aspvall, M. Plass, and R. Tarjan,
// A Linear-Time Algorithm for Testing the Truth of Certain Quantified Boolean
// Formulas
struct two_sat {
  public:
    two_sat() : _n(0), scc(0) {}
    two_sat(int n) : _n(n), _answer(n), scc(2 * n) {}

    void add_clause(int i, bool f, int j, bool g) {
        assert(0 <= i && i < _n);
        assert(0 <= j && j < _n);
        scc.add_edge(2 * i + (f ? 0 : 1), 2 * j + (g ? 1 : 0));
        scc.add_edge(2 * j + (g ? 0 : 1), 2 * i + (f ? 1 : 0));
    }
    bool satisfiable() {
        auto id = scc.scc_ids().second;
        for (int i = 0; i < _n; i++) {
            if (id[2 * i] == id[2 * i + 1]) return false;
            _answer[i] = id[2 * i] < id[2 * i + 1];
        }
        return true;
    }
    std::vector<bool> answer() { return _answer; }

//  private:
    int _n;
    std::vector<bool> _answer;
    internal::scc_graph scc;
};

}  // namespace atcoder

#endif  // ATCODER_TWOSAT_HPP

// <<< AtCoder <<<

using namespace std;
using namespace atcoder;
#define PARSE_ARGS(types, ...) if(!PyArg_ParseTuple(args, types, __VA_ARGS__)) return NULL


struct TwoSAT{
    PyObject_HEAD
    two_sat* ts;
};


extern PyTypeObject TwoSATType;


// >>> TwoSAT definition >>>

static void TwoSAT_dealloc(TwoSAT* self){
    delete self->ts;
    Py_TYPE(self)->tp_free((PyObject*)self);
}
static PyObject* TwoSAT_new(PyTypeObject* type, PyObject* args, PyObject* kwds){
    return type->tp_alloc(type, 0);
}
static int TwoSAT_init(TwoSAT* self, PyObject* args){
    long n;
    if(!PyArg_ParseTuple(args, "l", &n)) return -1;
    if(n < 0 || n > (long)1e8){
        PyErr_Format(PyExc_IndexError,
            "TwoSAT constructor constraint error (constraint: 0<=n<=1e8, got n=%d)", n);
        return -1;
    }
    self->ts = new two_sat(n);
    return 0;
}
static PyObject* TwoSAT_add_clause(TwoSAT* self, PyObject* args){
    long i, j;
    int f, g;
    PARSE_ARGS("lplp", &i, &f, &j, &g);
    if(i < 0 || i >= self->ts->_n || j < 0 || j >= self->ts->_n){
        PyErr_Format(PyExc_IndexError,
            "TwoSAT add_clause index out of range (n=%d, i=%d, j=%d)", self->ts->_n, i, j);
        return (PyObject*)NULL;
    }
    self->ts->add_clause(i, (bool)f, j, (bool)g);
    Py_RETURN_NONE;
}
static PyObject* TwoSAT_satisfiable(TwoSAT* self, PyObject* args){
    PyObject* res = self->ts->satisfiable() ? Py_True : Py_False;
    return Py_BuildValue("O", res);
}
static PyObject* TwoSAT_answer(TwoSAT* self, PyObject* args){
    const vector<bool>& answer = self->ts->answer();
    const int& n = self->ts->_n;
    PyObject* list = PyList_New(n);
    for(int i = 0; i < n; i++){
        PyList_SET_ITEM(list, i, Py_BuildValue("O", answer[i] ? Py_True : Py_False));
    }
    return list;
}
/*
static PyObject* TwoSAT_repr(PyObject* self){
    PyObject* res = PyUnicode_FromFormat("TwoSAT()");
    return res;
}
*/
static PyMethodDef TwoSAT_methods[] = {
    {"add_clause", (PyCFunction)TwoSAT_add_clause, METH_VARARGS, "Add clause"},
    {"satisfiable", (PyCFunction)TwoSAT_satisfiable, METH_VARARGS, "Check if problem satisfiable"},
    {"answer", (PyCFunction)TwoSAT_answer, METH_VARARGS, "Get answer"},
    {NULL}  /* Sentinel */
};
PyTypeObject TwoSATType = {
    PyObject_HEAD_INIT(NULL)
    "acl_twosat.TwoSAT",                   /*tp_name*/
    sizeof(TwoSAT),                     /*tp_basicsize*/
    0,                                  /*tp_itemsize*/
    (destructor)TwoSAT_dealloc,         /*tp_dealloc*/
    0,                                  /*tp_print*/
    0,                                  /*tp_getattr*/
    0,                                  /*tp_setattr*/
    0,                                  /*reserved*/
    0,//TwoSAT_repr,                      /*tp_repr*/
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
    TwoSAT_methods,                     /*tp_methods*/
    0,                                  /*tp_members*/
    0,                                  /*tp_getset*/
    0,                                  /*tp_base*/
    0,                                  /*tp_dict*/
    0,                                  /*tp_descr_get*/
    0,                                  /*tp_descr_set*/
    0,                                  /*tp_dictoffset*/
    (initproc)TwoSAT_init,              /*tp_init*/
    0,                                  /*tp_alloc*/
    TwoSAT_new,                         /*tp_new*/
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

// <<< TwoSAT definition <<<


static PyModuleDef acl_twosatmodule = {
    PyModuleDef_HEAD_INIT,
    "acl_twosat",
    NULL,
    -1,
};

PyMODINIT_FUNC PyInit_acl_twosat(void)
{
    PyObject* m;
    if(PyType_Ready(&TwoSATType) < 0) return NULL;

    m = PyModule_Create(&acl_twosatmodule);
    if(m == NULL) return NULL;

    Py_INCREF(&TwoSATType);
    if (PyModule_AddObject(m, "TwoSAT", (PyObject*)&TwoSATType) < 0) {
        Py_DECREF(&TwoSATType);
        Py_DECREF(m);
        return NULL;
    }
    
    return m;
}
"""
code_two_sat_setup = r"""
from distutils.core import setup, Extension
module = Extension(
    "acl_twosat",
    sources=["two_sat.cpp"],
    extra_compile_args=["-O3", "-march=native", "-std=c++14"]
)
setup(
    name="acl_twosat",
    version="0.0.1",
    description="wrapper for atcoder library twosat",
    ext_modules=[module]
)
"""

import os
import sys

if sys.argv[-1] == "ONLINE_JUDGE" or os.getcwd() != "/imojudge/sandbox":
    with open("two_sat.cpp", "w") as f:
        f.write(code_two_sat)
    with open("two_sat_setup.py", "w") as f:
        f.write(code_two_sat_setup)
    os.system(f"{sys.executable} two_sat_setup.py build_ext --inplace")

from acl_twosat import TwoSAT
