# TODO: メモリリーク確認
# TODO: min_cut, change_edge が正しく動くか確認

code_mf_graph = r"""
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

// 元のライブラリの private を剥がした

// >>> AtCoder >>>

#ifndef ATCODER_MAXFLOW_HPP
#define ATCODER_MAXFLOW_HPP 1

#include <algorithm>

#ifndef ATCODER_INTERNAL_QUEUE_HPP
#define ATCODER_INTERNAL_QUEUE_HPP 1

#include <vector>

namespace atcoder {

namespace internal {

template <class T> struct simple_queue {
    std::vector<T> payload;
    int pos = 0;
    void reserve(int n) { payload.reserve(n); }
    int size() const { return int(payload.size()) - pos; }
    bool empty() const { return pos == int(payload.size()); }
    void push(const T& t) { payload.push_back(t); }
    T& front() { return payload[pos]; }
    void clear() {
        payload.clear();
        pos = 0;
    }
    void pop() { pos++; }
};

}  // namespace internal

}  // namespace atcoder

#endif  // ATCODER_INTERNAL_QUEUE_HPP

#include <cassert>
#include <limits>
#include <queue>
#include <vector>

namespace atcoder {

template <class Cap> struct mf_graph {
  public:
    mf_graph() : _n(0) {}
    mf_graph(int n) : _n(n), g(n) {}

    int add_edge(int from, int to, Cap cap) {
        assert(0 <= from && from < _n);
        assert(0 <= to && to < _n);
        assert(0 <= cap);
        int m = int(pos.size());
        pos.push_back({from, int(g[from].size())});
        int from_id = int(g[from].size());
        int to_id = int(g[to].size());
        if (from == to) to_id++;
        g[from].push_back(_edge{to, to_id, cap});
        g[to].push_back(_edge{from, from_id, 0});
        return m;
    }

    struct edge {
        int from, to;
        Cap cap, flow;
    };

    edge get_edge(int i) {
        int m = int(pos.size());
        assert(0 <= i && i < m);
        auto _e = g[pos[i].first][pos[i].second];
        auto _re = g[_e.to][_e.rev];
        return edge{pos[i].first, _e.to, _e.cap + _re.cap, _re.cap};
    }
    std::vector<edge> edges() {
        int m = int(pos.size());
        std::vector<edge> result;
        for (int i = 0; i < m; i++) {
            result.push_back(get_edge(i));
        }
        return result;
    }
    void change_edge(int i, Cap new_cap, Cap new_flow) {
        int m = int(pos.size());
        assert(0 <= i && i < m);
        assert(0 <= new_flow && new_flow <= new_cap);
        auto& _e = g[pos[i].first][pos[i].second];
        auto& _re = g[_e.to][_e.rev];
        _e.cap = new_cap - new_flow;
        _re.cap = new_flow;
    }

    Cap flow(int s, int t) {
        return flow(s, t, std::numeric_limits<Cap>::max());
    }
    Cap flow(int s, int t, Cap flow_limit) {
        assert(0 <= s && s < _n);
        assert(0 <= t && t < _n);
        assert(s != t);

        std::vector<int> level(_n), iter(_n);
        internal::simple_queue<int> que;

        auto bfs = [&]() {
            std::fill(level.begin(), level.end(), -1);
            level[s] = 0;
            que.clear();
            que.push(s);
            while (!que.empty()) {
                int v = que.front();
                que.pop();
                for (auto e : g[v]) {
                    if (e.cap == 0 || level[e.to] >= 0) continue;
                    level[e.to] = level[v] + 1;
                    if (e.to == t) return;
                    que.push(e.to);
                }
            }
        };
        auto dfs = [&](auto self, int v, Cap up) {
            if (v == s) return up;
            Cap res = 0;
            int level_v = level[v];
            for (int& i = iter[v]; i < int(g[v].size()); i++) {
                _edge& e = g[v][i];
                if (level_v <= level[e.to] || g[e.to][e.rev].cap == 0) continue;
                Cap d =
                    self(self, e.to, std::min(up - res, g[e.to][e.rev].cap));
                if (d <= 0) continue;
                g[v][i].cap += d;
                g[e.to][e.rev].cap -= d;
                res += d;
                if (res == up) break;
            }
            return res;
        };

        Cap flow = 0;
        while (flow < flow_limit) {
            bfs();
            if (level[t] == -1) break;
            std::fill(iter.begin(), iter.end(), 0);
            while (flow < flow_limit) {
                Cap f = dfs(dfs, t, flow_limit - flow);
                if (!f) break;
                flow += f;
            }
        }
        return flow;
    }

    std::vector<bool> min_cut(int s) {
        std::vector<bool> visited(_n);
        internal::simple_queue<int> que;
        que.push(s);
        while (!que.empty()) {
            int p = que.front();
            que.pop();
            visited[p] = true;
            for (auto e : g[p]) {
                if (e.cap && !visited[e.to]) {
                    visited[e.to] = true;
                    que.push(e.to);
                }
            }
        }
        return visited;
    }

//  private:
    int _n;
    struct _edge {
        int to, rev;
        Cap cap;
    };
    std::vector<std::pair<int, int>> pos;
    std::vector<std::vector<_edge>> g;
};

}  // namespace atcoder

#endif  // ATCODER_MAXFLOW_HPP

// <<< AtCoder <<<

using namespace std;
using namespace atcoder;
#define PARSE_ARGS(types, ...) if(!PyArg_ParseTuple(args, types, __VA_ARGS__)) return NULL


struct MFGraph{
    PyObject_HEAD
    mf_graph<long long>* graph;
    //unique_ptr<mf_graph<long long>> graph;
};

struct MFGraphEdge{
    PyObject_HEAD
    mf_graph<long long>::edge* edge;
    //unique_ptr<mf_graph<long long>::edge> edge;
};


extern PyTypeObject MFGraphType;
extern PyTypeObject MFGraphEdgeType;


// >>> MFGraph definition >>>

static void MFGraph_dealloc(MFGraph* self){
    delete self->graph;
    Py_TYPE(self)->tp_free((PyObject*)self);
}
static PyObject* MFGraph_new(PyTypeObject* type, PyObject* args, PyObject* kwds){
    MFGraph* self;
    self = (MFGraph*)type->tp_alloc(type, 0);
    return (PyObject*)self;
}
static int MFGraph_init(MFGraph* self, PyObject* args){
    long n;
    if(!PyArg_ParseTuple(args, "l", &n)) return -1;
    if(n < 0 || n > (long)1e8){
        PyErr_Format(PyExc_IndexError,
            "constraint error in MFGraph constructor (constraint: 0<=n<=1e8, got n=%d)", n);
        return -1;
    }
    //self->graph = make_unique<mf_graph<long long>>(n);
    self->graph = new mf_graph<long long>(n);
    return 0;
}
static PyObject* MFGraph_add_edge(MFGraph* self, PyObject* args){
    long from, to;
    long long cap;
    PARSE_ARGS("llL", &from, &to, &cap);
    if(from < 0 || from >= self->graph->_n || to < 0 || to >= self->graph->_n){
        PyErr_Format(PyExc_IndexError,
            "MFGraph add_edge index out of range (n=%d, from=%d, to=%d)", self->graph->_n, from, to);
        return (PyObject*)NULL;
    }
    if(from == to){
        PyErr_Format(PyExc_IndexError, "got self-loop (from=%d, to=%d)", from, to);
        return (PyObject*)NULL;
    }
    if(cap < 0){
        PyErr_Format(PyExc_IndexError, "got negative cap (cap=%d)", cap);
        return (PyObject*)NULL;
    }
    const int res = self->graph->add_edge(from, to, cap);
    return Py_BuildValue("l", res);
}
static PyObject* MFGraph_flow(MFGraph* self, PyObject* args){
    long s, t;
    long long flow_limit = numeric_limits<long long>::max();
    PARSE_ARGS("ll|L", &s, &t, &flow_limit);
    if(s < 0 || s >= self->graph->_n || t < 0 || t >= self->graph->_n){
        PyErr_Format(PyExc_IndexError,
            "MFGraph flow index out of range (n=%d, s=%d, t=%d)", self->graph->_n, s, t);
        return (PyObject*)NULL;
    }
    if(s == t){
        PyErr_Format(PyExc_IndexError, "got s == t (s=%d, t=%d)", s, t);
        return (PyObject*)NULL;
    }
    const long long& flow = self->graph->flow(s, t, flow_limit);
    return Py_BuildValue("L", flow);
}
static PyObject* MFGraph_min_cut(MFGraph* self, PyObject* args){
    long s;
    PARSE_ARGS("l", &s);
    if(s < 0 || s >= self->graph->_n){
        PyErr_Format(PyExc_IndexError,
            "MFGraph min_cut index out of range (n=%d, s=%d)", self->graph->_n, s);
        return (PyObject*)NULL;
    }
    const vector<bool>& vec = self->graph->min_cut(s);
    PyObject* list = PyList_New(vec.size());
    for(int i = 0; i < (int)vec.size(); i++){
        PyObject* b = vec[i] ? Py_True : Py_False;
        Py_INCREF(b);
        PyList_SET_ITEM(list, i, b);
    }
    return list;
}
static PyObject* MFGraph_get_edge(MFGraph* self, PyObject* args){
    long i;
    PARSE_ARGS("l", &i);
    const int m = (int)self->graph->pos.size();
    if(i < 0 || i >= m){
        PyErr_Format(PyExc_IndexError,
            "MFGraph get_edge index out of range (m=%d, i=%d)", m, i);
        return (PyObject*)NULL;
    }
    MFGraphEdge* edge = PyObject_NEW(MFGraphEdge, &MFGraphEdgeType);
    //edge->edge = make_unique<mf_graph<long long>::edge>(self->graph->get_edge(i));  // なぜか edge に値が入っていて詰まる
    edge->edge = new mf_graph<long long>::edge(self->graph->get_edge(i));
    return (PyObject*)edge;
}
static PyObject* MFGraph_edges(MFGraph* self, PyObject* args){
    const auto& edges = self->graph->edges();
    const int m = (int)edges.size();
    PyObject* list = PyList_New(m);
    for(int i = 0; i < m; i++){
        MFGraphEdge* edge = PyObject_NEW(MFGraphEdge, &MFGraphEdgeType);
        //edge->edge = make_unique<mf_graph<long long>::edge>(edges[i]);
        edge->edge = new mf_graph<long long>::edge(edges[i]);
        PyList_SET_ITEM(list, i, (PyObject*)edge);
    }
    return list;
}
static PyObject* MFGraph_change_edge(MFGraph* self, PyObject* args){
    long i;
    long long new_cap, new_flow;
    PARSE_ARGS("lLL", &i, &new_cap, &new_flow);
    const int m = (int)self->graph->pos.size();
    if(i < 0 || i >= m){
        PyErr_Format(PyExc_IndexError,
            "MFGraph change_edge index out of range (m=%d, i=%d)", m, i);
        return (PyObject*)NULL;
    }
    if(new_flow < 0 || new_cap < new_flow){
        PyErr_Format(
            PyExc_IndexError,
            "MFGraph change_edge constraint error (constraint: 0<=new_flow<=new_cap, got new_flow=%lld, new_cap=%lld)",
            new_flow, new_cap
        );
        return (PyObject*)NULL;
    }
    self->graph->change_edge(i, new_cap, new_flow);
    Py_RETURN_NONE;
}
static PyObject* MFGraph_repr(PyObject* self){
    PyObject* edges = MFGraph_edges((MFGraph*)self, NULL);
    PyObject* res = PyUnicode_FromFormat("MFGraph(%R)", edges);
    Py_DECREF(edges);
    return res;
}
static PyMethodDef MFGraph_methods[] = {
    {"add_edge", (PyCFunction)MFGraph_add_edge, METH_VARARGS, "Add edge"},
    {"flow", (PyCFunction)MFGraph_flow, METH_VARARGS, "Flow"},
    {"min_cut", (PyCFunction)MFGraph_min_cut, METH_VARARGS, "Get vertices those can be reached from source"},
    {"get_edge", (PyCFunction)MFGraph_get_edge, METH_VARARGS, "Get edge"},
    {"edges", (PyCFunction)MFGraph_edges, METH_VARARGS, "Get edges"},
    {"change_edge", (PyCFunction)MFGraph_change_edge, METH_VARARGS, "Change edge"},
    {NULL}  /* Sentinel */
};
PyTypeObject MFGraphType = {
    PyObject_HEAD_INIT(NULL)
    "atcoder.MFGraph",                  /*tp_name*/
    sizeof(MFGraph),                    /*tp_basicsize*/
    0,                                  /*tp_itemsize*/
    (destructor)MFGraph_dealloc,        /*tp_dealloc*/
    0,                                  /*tp_print*/
    0,                                  /*tp_getattr*/
    0,                                  /*tp_setattr*/
    0,                                  /*reserved*/
    MFGraph_repr,                       /*tp_repr*/
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
    MFGraph_methods,                    /*tp_methods*/
    0,                                  /*tp_members*/
    0,                                  /*tp_getset*/
    0,                                  /*tp_base*/
    0,                                  /*tp_dict*/
    0,                                  /*tp_descr_get*/
    0,                                  /*tp_descr_set*/
    0,                                  /*tp_dictoffset*/
    (initproc)MFGraph_init,             /*tp_init*/
    0,                                  /*tp_alloc*/
    MFGraph_new,                        /*tp_new*/
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

// <<< MFGraph definition <<<


// >>> MFGraphEdge definition >>>

static void MFGraphEdge_dealloc(MFGraphEdge* self){
    delete self->edge;
    Py_TYPE(self)->tp_free((PyObject*)self);
}
static PyObject* MFGraphEdge_new(PyTypeObject* type, PyObject* args, PyObject* kwds){
    //MFGraphEdge* self;
    //self = (MFGraphEdge*)type->tp_alloc(type, 0);
    //return (PyObject*)self;
    return type->tp_alloc(type, 0);
}
static int MFGraphEdge_init(MFGraphEdge* self, PyObject* args){
    int from, to;
    long long cap, flow;
    if(!PyArg_ParseTuple(args, "llLL", &from, &to, &cap, &flow)) return -1;
    //self->edge = make_unique<mf_graph<long long>::edge>(mf_graph<long long>::edge{from, to, cap, flow});
    self->edge = new mf_graph<long long>::edge(mf_graph<long long>::edge{from, to, cap, flow});
    return 0;
}
static PyObject* MFGraphEdge_get_from(MFGraphEdge* self, PyObject* args){
    return PyLong_FromLong(self->edge->from);
}
static PyObject* MFGraphEdge_get_to(MFGraphEdge* self, PyObject* args){
    return PyLong_FromLong(self->edge->to);
}
static PyObject* MFGraphEdge_get_flow(MFGraphEdge* self, PyObject* args){
    return PyLong_FromLongLong(self->edge->flow);
}
static PyObject* MFGraphEdge_get_cap(MFGraphEdge* self, PyObject* args){
    return PyLong_FromLongLong(self->edge->cap);
}
static PyObject* MFGraphEdge_repr(PyObject* self){
    MFGraphEdge* self_ = (MFGraphEdge*)self;
    PyObject* res = PyUnicode_FromFormat("MFGraphEdge(%2d -> %2d, %2lld / %2lld)",
        self_->edge->from, self_->edge->to, self_->edge->flow, self_->edge->cap);
    return res;
}
PyGetSetDef MFGraphEdge_getsets[] = {
    {"from_", (getter)MFGraphEdge_get_from, NULL, NULL, NULL},
    {"to", (getter)MFGraphEdge_get_to, NULL, NULL, NULL},
    {"flow", (getter)MFGraphEdge_get_flow, NULL, NULL, NULL},
    {"cap", (getter)MFGraphEdge_get_cap, NULL, NULL, NULL},
    {NULL}
};
PyTypeObject MFGraphEdgeType = {
    PyObject_HEAD_INIT(NULL)
    "atcoder.MFGraphEdge",              /*tp_name*/
    sizeof(MFGraphEdge),                /*tp_basicsize*/
    0,                                  /*tp_itemsize*/
    (destructor)MFGraphEdge_dealloc,    /*tp_dealloc*/
    0,                                  /*tp_print*/
    0,                                  /*tp_getattr*/
    0,                                  /*tp_setattr*/
    0,                                  /*reserved*/
    MFGraphEdge_repr,                   /*tp_repr*/
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
    0,                                  /*tp_methods*/
    0,                                  /*tp_members*/
    MFGraphEdge_getsets,                /*tp_getset*/
    0,                                  /*tp_base*/
    0,                                  /*tp_dict*/
    0,                                  /*tp_descr_get*/
    0,                                  /*tp_descr_set*/
    0,                                  /*tp_dictoffset*/
    (initproc)MFGraphEdge_init,         /*tp_init*/
    0,                                  /*tp_alloc*/
    MFGraphEdge_new,                    /*tp_new*/
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

// <<< MFGraphEdge definition <<<


static PyModuleDef atcodermodule = {
    PyModuleDef_HEAD_INIT,
    "atcoder",
    NULL,
    -1,
};

PyMODINIT_FUNC PyInit_atcoder(void)
{
    PyObject* m;
    if(PyType_Ready(&MFGraphType) < 0) return NULL;
    if(PyType_Ready(&MFGraphEdgeType) < 0) return NULL;

    m = PyModule_Create(&atcodermodule);
    if(m == NULL) return NULL;

    Py_INCREF(&MFGraphType);
    if (PyModule_AddObject(m, "MFGraph", (PyObject*)&MFGraphType) < 0) {
        Py_DECREF(&MFGraphType);
        Py_DECREF(m);
        return NULL;
    }
    
    Py_INCREF(&MFGraphEdgeType);
    if (PyModule_AddObject(m, "MFGraphEdge", (PyObject*)&MFGraphEdgeType) < 0) {
        Py_DECREF(&MFGraphEdgeType);
        Py_DECREF(m);
        return NULL;
    }
    return m;
}
"""
code_mf_graph_setup = r"""
from distutils.core import setup, Extension
module = Extension(
    "atcoder",
    sources=["mf_graph.cpp"],
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
    with open("mf_graph.cpp", "w") as f:
        f.write(code_mf_graph)
    with open("mf_graph_setup.py", "w") as f:
        f.write(code_mf_graph_setup)
    os.system(f"{sys.executable} mf_graph_setup.py build_ext --inplace")

from atcoder import MFGraph, MFGraphEdge

