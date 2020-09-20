# TODO: メモリリーク確認

code_mcf_graph = r"""
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

// 元のライブラリの private を剥がした

// >>> AtCoder >>>

#ifndef ATCODER_MINCOSTFLOW_HPP
#define ATCODER_MINCOSTFLOW_HPP 1

#include <algorithm>
#include <cassert>
#include <limits>
#include <queue>
#include <vector>

namespace atcoder {

template <class Cap, class Cost> struct mcf_graph {
  public:
    mcf_graph() {}
    mcf_graph(int n) : _n(n), g(n) {}

    int add_edge(int from, int to, Cap cap, Cost cost) {
        assert(0 <= from && from < _n);
        assert(0 <= to && to < _n);
        int m = int(pos.size());
        pos.push_back({from, int(g[from].size())});
        int from_id = int(g[from].size());
        int to_id = int(g[to].size());
        if (from == to) to_id++;
        g[from].push_back(_edge{to, to_id, cap, cost});
        g[to].push_back(_edge{from, from_id, 0, -cost});
        return m;
    }

    struct edge {
        int from, to;
        Cap cap, flow;
        Cost cost;
    };

    edge get_edge(int i) {
        int m = int(pos.size());
        assert(0 <= i && i < m);
        auto _e = g[pos[i].first][pos[i].second];
        auto _re = g[_e.to][_e.rev];
        return edge{
            pos[i].first, _e.to, _e.cap + _re.cap, _re.cap, _e.cost,
        };
    }
    std::vector<edge> edges() {
        int m = int(pos.size());
        std::vector<edge> result(m);
        for (int i = 0; i < m; i++) {
            result[i] = get_edge(i);
        }
        return result;
    }

    std::pair<Cap, Cost> flow(int s, int t) {
        return flow(s, t, std::numeric_limits<Cap>::max());
    }
    std::pair<Cap, Cost> flow(int s, int t, Cap flow_limit) {
        return slope(s, t, flow_limit).back();
    }
    std::vector<std::pair<Cap, Cost>> slope(int s, int t) {
        return slope(s, t, std::numeric_limits<Cap>::max());
    }
    std::vector<std::pair<Cap, Cost>> slope(int s, int t, Cap flow_limit) {
        assert(0 <= s && s < _n);
        assert(0 <= t && t < _n);
        assert(s != t);
        // variants (C = maxcost):
        // -(n-1)C <= dual[s] <= dual[i] <= dual[t] = 0
        // reduced cost (= e.cost + dual[e.from] - dual[e.to]) >= 0 for all edge
        std::vector<Cost> dual(_n, 0), dist(_n);
        std::vector<int> pv(_n), pe(_n);
        std::vector<bool> vis(_n);
        auto dual_ref = [&]() {
            std::fill(dist.begin(), dist.end(),
                      std::numeric_limits<Cost>::max());
            std::fill(pv.begin(), pv.end(), -1);
            std::fill(pe.begin(), pe.end(), -1);
            std::fill(vis.begin(), vis.end(), false);
            struct Q {
                Cost key;
                int to;
                bool operator<(Q r) const { return key > r.key; }
            };
            std::priority_queue<Q> que;
            dist[s] = 0;
            que.push(Q{0, s});
            while (!que.empty()) {
                int v = que.top().to;
                que.pop();
                if (vis[v]) continue;
                vis[v] = true;
                if (v == t) break;
                // dist[v] = shortest(s, v) + dual[s] - dual[v]
                // dist[v] >= 0 (all reduced cost are positive)
                // dist[v] <= (n-1)C
                for (int i = 0; i < int(g[v].size()); i++) {
                    auto e = g[v][i];
                    if (vis[e.to] || !e.cap) continue;
                    // |-dual[e.to] + dual[v]| <= (n-1)C
                    // cost <= C - -(n-1)C + 0 = nC
                    Cost cost = e.cost - dual[e.to] + dual[v];
                    if (dist[e.to] - dist[v] > cost) {
                        dist[e.to] = dist[v] + cost;
                        pv[e.to] = v;
                        pe[e.to] = i;
                        que.push(Q{dist[e.to], e.to});
                    }
                }
            }
            if (!vis[t]) {
                return false;
            }

            for (int v = 0; v < _n; v++) {
                if (!vis[v]) continue;
                // dual[v] = dual[v] - dist[t] + dist[v]
                //         = dual[v] - (shortest(s, t) + dual[s] - dual[t]) + (shortest(s, v) + dual[s] - dual[v])
                //         = - shortest(s, t) + dual[t] + shortest(s, v)
                //         = shortest(s, v) - shortest(s, t) >= 0 - (n-1)C
                dual[v] -= dist[t] - dist[v];
            }
            return true;
        };
        Cap flow = 0;
        Cost cost = 0, prev_cost_per_flow = -1;
        std::vector<std::pair<Cap, Cost>> result;
        result.push_back({flow, cost});
        while (flow < flow_limit) {
            if (!dual_ref()) break;
            Cap c = flow_limit - flow;
            for (int v = t; v != s; v = pv[v]) {
                c = std::min(c, g[pv[v]][pe[v]].cap);
            }
            for (int v = t; v != s; v = pv[v]) {
                auto& e = g[pv[v]][pe[v]];
                e.cap -= c;
                g[v][e.rev].cap += c;
            }
            Cost d = -dual[s];
            flow += c;
            cost += c * d;
            if (prev_cost_per_flow == d) {
                result.pop_back();
            }
            result.push_back({flow, cost});
            prev_cost_per_flow = d;
        }
        return result;
    }

//  private:
    int _n;

    struct _edge {
        int to, rev;
        Cap cap;
        Cost cost;
    };

    std::vector<std::pair<int, int>> pos;
    std::vector<std::vector<_edge>> g;
};

}  // namespace atcoder

#endif  // ATCODER_MINCOSTFLOW_HPP

// <<< AtCoder <<<

using namespace std;
using namespace atcoder;
#define PARSE_ARGS(types, ...) if(!PyArg_ParseTuple(args, types, __VA_ARGS__)) return NULL


struct MCFGraph{
    PyObject_HEAD
    mcf_graph<long long, long long>* graph;
};

struct MCFGraphEdge{
    PyObject_HEAD
    mcf_graph<long long, long long>::edge* edge;
};


extern PyTypeObject MCFGraphType;
extern PyTypeObject MCFGraphEdgeType;


// >>> MCFGraph definition >>>

static void MCFGraph_dealloc(MCFGraph* self){
    delete self->graph;
    Py_TYPE(self)->tp_free((PyObject*)self);
}
static PyObject* MCFGraph_new(PyTypeObject* type, PyObject* args, PyObject* kwds){
    return type->tp_alloc(type, 0);
}
static int MCFGraph_init(MCFGraph* self, PyObject* args){
    long n;
    if(!PyArg_ParseTuple(args, "l", &n)) return -1;
    if(n < 0 || n > (long)1e8){
        PyErr_Format(PyExc_IndexError,
            "constraint error in MCFGraph constructor (constraint: 0<=n<=1e8, got n=%d)", n);
        return -1;
    }
    self->graph = new mcf_graph<long long, long long>(n);
    return 0;
}
static PyObject* MCFGraph_add_edge(MCFGraph* self, PyObject* args){
    long from, to;
    long long cap, cost;
    PARSE_ARGS("llLL", &from, &to, &cap, &cost);
    if(from < 0 || from >= self->graph->_n || to < 0 || to >= self->graph->_n){
        PyErr_Format(PyExc_IndexError,
            "MCFGraph add_edge index out of range (n=%d, from=%d, to=%d)", self->graph->_n, from, to);
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
    if(cost < 0){
        PyErr_Format(PyExc_IndexError, "got negative cost (cap=%d)", cost);
        return (PyObject*)NULL;
    }
    const int res = self->graph->add_edge(from, to, cap, cost);
    return Py_BuildValue("l", res);
}
static PyObject* MCFGraph_flow(MCFGraph* self, PyObject* args){
    long s, t;
    long long flow_limit = numeric_limits<long long>::max();
    PARSE_ARGS("ll|L", &s, &t, &flow_limit);
    if(s < 0 || s >= self->graph->_n || t < 0 || t >= self->graph->_n){
        PyErr_Format(PyExc_IndexError,
            "MCFGraph flow index out of range (n=%d, s=%d, t=%d)", self->graph->_n, s, t);
        return (PyObject*)NULL;
    }
    if(s == t){
        PyErr_Format(PyExc_IndexError, "got s == t (s=%d, t=%d)", s, t);
        return (PyObject*)NULL;
    }
    const pair<long long, long long>& flow_cost = self->graph->flow(s, t, flow_limit);
    return Py_BuildValue("LL", flow_cost.first, flow_cost.second);
}
static PyObject* MCFGraph_slope(MCFGraph* self, PyObject* args){
    long s, t;
    long long flow_limit = numeric_limits<long long>::max();
    PARSE_ARGS("ll|L", &s, &t, &flow_limit);
    if(s < 0 || s >= self->graph->_n || t < 0 || t >= self->graph->_n){
        PyErr_Format(PyExc_IndexError,
            "MCFGraph slope index out of range (n=%d, s=%d, t=%d)", self->graph->_n, s, t);
        return (PyObject*)NULL;
    }
    if(s == t){
        PyErr_Format(PyExc_IndexError, "got s == t (s=%d, t=%d)", s, t);
        return (PyObject*)NULL;
    }
    const vector<pair<long long, long long>>& slope = self->graph->slope(s, t, flow_limit);
    const int siz = (int)slope.size();
    PyObject* list = PyList_New(siz);
    for(int i = 0; i < siz; i++){
        PyList_SET_ITEM(list, i, Py_BuildValue("LL", slope[i].first, slope[i].second));
    }
    return list;
}
static PyObject* MCFGraph_get_edge(MCFGraph* self, PyObject* args){
    long i;
    PARSE_ARGS("l", &i);
    const int m = (int)self->graph->pos.size();
    if(i < 0 || i >= m){
        PyErr_Format(PyExc_IndexError,
            "MCFGraph get_edge index out of range (m=%d, i=%d)", m, i);
        return (PyObject*)NULL;
    }
    MCFGraphEdge* edge = PyObject_NEW(MCFGraphEdge, &MCFGraphEdgeType);
    edge->edge = new mcf_graph<long long, long long>::edge(self->graph->get_edge(i));
    return (PyObject*)edge;
}
static PyObject* MCFGraph_edges(MCFGraph* self, PyObject* args){
    const auto& edges = self->graph->edges();
    const int m = (int)edges.size();
    PyObject* list = PyList_New(m);
    for(int i = 0; i < m; i++){
        MCFGraphEdge* edge = PyObject_NEW(MCFGraphEdge, &MCFGraphEdgeType);
        edge->edge = new mcf_graph<long long, long long>::edge(edges[i]);
        PyList_SET_ITEM(list, i, (PyObject*)edge);
    }
    return list;
}
static PyObject* MCFGraph_repr(PyObject* self){
    PyObject* edges = MCFGraph_edges((MCFGraph*)self, NULL);
    PyObject* res = PyUnicode_FromFormat("MCFGraph(%R)", edges);
    Py_DECREF(edges);
    return res;
}
static PyMethodDef MCFGraph_methods[] = {
    {"add_edge", (PyCFunction)MCFGraph_add_edge, METH_VARARGS, "Add edge"},
    {"flow", (PyCFunction)MCFGraph_flow, METH_VARARGS, "Flow"},
    {"slope", (PyCFunction)MCFGraph_slope, METH_VARARGS, "Slope"},
    {"get_edge", (PyCFunction)MCFGraph_get_edge, METH_VARARGS, "Get edge"},
    {"edges", (PyCFunction)MCFGraph_edges, METH_VARARGS, "Get edges"},
    {NULL}  /* Sentinel */
};
PyTypeObject MCFGraphType = {
    PyObject_HEAD_INIT(NULL)
    "atcoder.MCFGraph",                 /*tp_name*/
    sizeof(MCFGraph),                   /*tp_basicsize*/
    0,                                  /*tp_itemsize*/
    (destructor)MCFGraph_dealloc,       /*tp_dealloc*/
    0,                                  /*tp_print*/
    0,                                  /*tp_getattr*/
    0,                                  /*tp_setattr*/
    0,                                  /*reserved*/
    MCFGraph_repr,                      /*tp_repr*/
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
    MCFGraph_methods,                   /*tp_methods*/
    0,                                  /*tp_members*/
    0,                                  /*tp_getset*/
    0,                                  /*tp_base*/
    0,                                  /*tp_dict*/
    0,                                  /*tp_descr_get*/
    0,                                  /*tp_descr_set*/
    0,                                  /*tp_dictoffset*/
    (initproc)MCFGraph_init,            /*tp_init*/
    0,                                  /*tp_alloc*/
    MCFGraph_new,                       /*tp_new*/
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

// <<< MCFGraph definition <<<


// >>> MCFGraphEdge definition >>>

static void MCFGraphEdge_dealloc(MCFGraphEdge* self){
    delete self->edge;
    Py_TYPE(self)->tp_free((PyObject*)self);
}
static PyObject* MCFGraphEdge_new(PyTypeObject* type, PyObject* args, PyObject* kwds){
    return type->tp_alloc(type, 0);
}
static int MCFGraphEdge_init(MCFGraphEdge* self, PyObject* args){
    int from, to;
    long long cap, flow, cost;
    if(!PyArg_ParseTuple(args, "llLLL", &from, &to, &cap, &flow, &cost)) return -1;
    self->edge = new mcf_graph<long long, long long>::edge(mcf_graph<long long, long long>::edge{from, to, cap, flow, cost});
    return 0;
}
static PyObject* MCFGraphEdge_get_from(MCFGraphEdge* self, PyObject* args){
    return PyLong_FromLong(self->edge->from);
}
static PyObject* MCFGraphEdge_get_to(MCFGraphEdge* self, PyObject* args){
    return PyLong_FromLong(self->edge->to);
}
static PyObject* MCFGraphEdge_get_flow(MCFGraphEdge* self, PyObject* args){
    return PyLong_FromLongLong(self->edge->flow);
}
static PyObject* MCFGraphEdge_get_cap(MCFGraphEdge* self, PyObject* args){
    return PyLong_FromLongLong(self->edge->cap);
}
static PyObject* MCFGraphEdge_get_cost(MCFGraphEdge* self, PyObject* args){
    return PyLong_FromLongLong(self->edge->cost);
}
static PyObject* MCFGraphEdge_repr(PyObject* self){
    MCFGraphEdge* self_ = (MCFGraphEdge*)self;
    PyObject* res = PyUnicode_FromFormat("MCFGraphEdge(%2d -> %2d, %2lld / %2lld, cost = %2lld)",
        self_->edge->from, self_->edge->to, self_->edge->flow, self_->edge->cap, self_->edge->cost);
    return res;
}
PyGetSetDef MCFGraphEdge_getsets[] = {
    {"from_", (getter)MCFGraphEdge_get_from, NULL, NULL, NULL},
    {"to", (getter)MCFGraphEdge_get_to, NULL, NULL, NULL},
    {"flow", (getter)MCFGraphEdge_get_flow, NULL, NULL, NULL},
    {"cap", (getter)MCFGraphEdge_get_cap, NULL, NULL, NULL},
    {"cost", (getter)MCFGraphEdge_get_cost, NULL, NULL, NULL},
    {NULL}
};
PyTypeObject MCFGraphEdgeType = {
    PyObject_HEAD_INIT(NULL)
    "atcoder.MCFGraphEdge",             /*tp_name*/
    sizeof(MCFGraphEdge),               /*tp_basicsize*/
    0,                                  /*tp_itemsize*/
    (destructor)MCFGraphEdge_dealloc,   /*tp_dealloc*/
    0,                                  /*tp_print*/
    0,                                  /*tp_getattr*/
    0,                                  /*tp_setattr*/
    0,                                  /*reserved*/
    MCFGraphEdge_repr,                  /*tp_repr*/
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
    MCFGraphEdge_getsets,               /*tp_getset*/
    0,                                  /*tp_base*/
    0,                                  /*tp_dict*/
    0,                                  /*tp_descr_get*/
    0,                                  /*tp_descr_set*/
    0,                                  /*tp_dictoffset*/
    (initproc)MCFGraphEdge_init,        /*tp_init*/
    0,                                  /*tp_alloc*/
    MCFGraphEdge_new,                   /*tp_new*/
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

// <<< MCFGraphEdge definition <<<


static PyModuleDef atcodermodule = {
    PyModuleDef_HEAD_INIT,
    "atcoder",
    NULL,
    -1,
};

PyMODINIT_FUNC PyInit_atcoder(void)
{
    PyObject* m;
    if(PyType_Ready(&MCFGraphType) < 0) return NULL;
    if(PyType_Ready(&MCFGraphEdgeType) < 0) return NULL;

    m = PyModule_Create(&atcodermodule);
    if(m == NULL) return NULL;

    Py_INCREF(&MCFGraphType);
    if (PyModule_AddObject(m, "MCFGraph", (PyObject*)&MCFGraphType) < 0) {
        Py_DECREF(&MCFGraphType);
        Py_DECREF(m);
        return NULL;
    }
    
    Py_INCREF(&MCFGraphEdgeType);
    if (PyModule_AddObject(m, "MCFGraphEdge", (PyObject*)&MCFGraphEdgeType) < 0) {
        Py_DECREF(&MCFGraphEdgeType);
        Py_DECREF(m);
        return NULL;
    }
    return m;
}
"""
code_mcf_graph_setup = r"""
from distutils.core import setup, Extension
module = Extension(
    "atcoder",
    sources=["mcf_graph.cpp"],
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
    with open("mcf_graph.cpp", "w") as f:
        f.write(code_mcf_graph)
    with open("mcf_graph_setup.py", "w") as f:
        f.write(code_mcf_graph_setup)
    os.system(f"{sys.executable} mcf_graph_setup.py build_ext --inplace")

from atcoder import MCFGraph, MCFGraphEdge
