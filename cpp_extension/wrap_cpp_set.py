code_cppset = r"""
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"
#include <vector>
//#undef __GNUC__  // g++ 拡張を使わない場合はここのコメントアウトを外すと高速になる
#ifdef __GNUC__
    #include <ext/pb_ds/assoc_container.hpp>
    #include <ext/pb_ds/tree_policy.hpp>
    using namespace std;
    using namespace __gnu_pbds;
    const static auto comp_pyobj = [](PyObject* const & lhs, PyObject* const & rhs){
        return (bool)PyObject_RichCompareBool(lhs, rhs, Py_LT);  // 比較できない場合 -1
    };
    using pb_set = tree<
        PyObject*,
        null_type,
        decltype(comp_pyobj),
        rb_tree_tag,
        tree_order_statistics_node_update
    >;
#else
    #include <set>
    using namespace std;
    const static auto comp_pyobj = [](PyObject* const & lhs, PyObject* const & rhs){
        return (bool)PyObject_RichCompareBool(lhs, rhs, Py_LT);
    };
    using pb_set = set<PyObject*, decltype(comp_pyobj)>;
#endif
#define PARSE_ARGS(types, ...) if(!PyArg_ParseTuple(args, types, __VA_ARGS__)) return NULL
struct Set4PyObject{
    pb_set st;
    pb_set::iterator it;
    Set4PyObject() : st(comp_pyobj), it(st.begin()) {}
    Set4PyObject(vector<PyObject*>& vec) : st(vec.begin(), vec.end(), comp_pyobj), it(st.begin()) {}
    Set4PyObject(const Set4PyObject& obj) : st(obj.st), it(st.begin()) {
        for(PyObject* const & p : st) Py_INCREF(p);
    }
    ~Set4PyObject(){
        for(PyObject* const & p : st) Py_DECREF(p);
    }
    bool add(PyObject* x){
        const auto& r = st.insert(x);
        it = r.first;
        if(r.second){
            Py_INCREF(x);
            return true;
        }else{
            return false;
        }
    }
    PyObject* remove(PyObject* x){
        it = st.find(x);
        if(it == st.end()) return PyErr_SetObject(PyExc_KeyError, x), (PyObject*)NULL;
        Py_DECREF(*it);
        it = st.erase(it);
        if(it == st.end()) return Py_None;
        return *it;
    }
    PyObject* search_higher_equal(PyObject* x){
        it = st.lower_bound(x);
        if(it == st.end()) return Py_None;
        return *it;
    }
    PyObject* min(){
        if(st.size()==0)
            return PyErr_SetString(PyExc_IndexError, "min from an empty set"), (PyObject*)NULL;
        it = st.begin();
        return *it;
    }
    PyObject* max(){
        if(st.size()==0)
            return PyErr_SetString(PyExc_IndexError, "min from an empty set"), (PyObject*)NULL;
        it = prev(st.end());
        return *it;
    }
    PyObject* pop_min(){
        if(st.size()==0)
            return PyErr_SetString(PyExc_IndexError, "pop_min from an empty set"), (PyObject*)NULL;
        it = st.begin();
        PyObject* res = *it;
        it = st.erase(it);
        return res;
    }
    PyObject* pop_max(){
        if(st.size()==0)
            return PyErr_SetString(PyExc_IndexError, "pop_max from an empty set"), (PyObject*)NULL;
        it = prev(st.end());
        PyObject* res = *it;
        it = st.erase(it);
        return res;
    }
    size_t len() const {
        return st.size();
    }
    PyObject* iter_next(){
        if(it == st.end()) return Py_None;
        if(++it == st.end()) return Py_None;
        return *it;
    }
    PyObject* iter_prev(){
        if(it == st.begin()) return Py_None;
        return *--it;
    }
    PyObject* to_list() const {
        PyObject* list = PyList_New(st.size());
        int i = 0;
        for(PyObject* const & p : st){
            Py_INCREF(p);
            PyList_SET_ITEM(list, i++, p);
        }
        return list;
    }
    PyObject* get() const {
        if(it == st.end()) return Py_None;
        return *it;
    }
    PyObject* erase(){
        if(it == st.end()) return PyErr_SetString(PyExc_KeyError, "erase end"), (PyObject*)NULL;
        it = st.erase(it);
        if(it == st.end()) return Py_None;
        return *it;
    }
    PyObject* getitem(const long& idx){
        long idx_pos = idx >= 0 ? idx : idx + (long)st.size();
        if(idx_pos >= (long)st.size() || idx_pos < 0)
            return PyErr_Format(
                PyExc_IndexError,
                "cppset getitem index out of range (size=%d, idx=%d)", st.size(), idx
            ), (PyObject*)NULL;
        #ifdef __GNUC__
            it = st.find_by_order(idx_pos);
        #else
            it = st.begin();
            for(int i=0; i<idx_pos; i++) it++;
        #endif
        return *it;
    }
    PyObject* pop(const long& idx){
        long idx_pos = idx >= 0 ? idx : idx + (long)st.size();
        if(idx_pos >= (long)st.size() || idx_pos < 0)
            return PyErr_Format(
                PyExc_IndexError,
                "cppset pop index out of range (size=%d, idx=%d)", st.size(), idx
            ), (PyObject*)NULL;
        #ifdef __GNUC__
            it = st.find_by_order(idx_pos);
        #else
            it = st.begin();
            for(int i=0; i<idx_pos; i++) it++;
        #endif
        PyObject* res = *it;
        it = st.erase(it);
        return res;
    }
    long index(PyObject* x) const {
        #ifdef __GNUC__
            return st.order_of_key(x);
        #else
            long res = 0;
            pb_set::iterator it2 = st.begin();
            while(it2 != st.end() && comp_pyobj(*it2, x)) it2++, res++;
            return res;
        #endif
    }
};

struct CppSet{
    PyObject_VAR_HEAD
    Set4PyObject* st;
};

extern PyTypeObject CppSetType;

static void CppSet_dealloc(CppSet* self){
    delete self->st;
    Py_TYPE(self)->tp_free((PyObject*)self);
}
static PyObject* CppSet_new(PyTypeObject* type, PyObject* args, PyObject* kwds){
    CppSet* self;
    self = (CppSet*)type->tp_alloc(type, 0);
    return (PyObject*)self;
}
static int CppSet_init(CppSet* self, PyObject* args, PyObject* kwds){
    static char* kwlist[] = {(char*)"lst", NULL};
    PyObject* lst = NULL;
    if(!PyArg_ParseTupleAndKeywords(args, kwds, "|O", kwlist, &lst)) return -1;
    if(lst == NULL){
        self->st = new Set4PyObject();
        Py_SIZE(self) = 0;
    }else{
        int siz;
        if(PyList_Check(lst)) siz = (int)PyList_GET_SIZE(lst);
        else if(PyTuple_Check(lst)) siz = (int)PyTuple_GET_SIZE(lst);
        else return PyErr_SetString(PyExc_TypeError, "got neither list nor tuple"), NULL;
        vector<PyObject*> vec(siz);
        for(int i=0; i<siz; i++){
            vec[i] = PyList_Check(lst) ? PyList_GET_ITEM(lst, i) : PyTuple_GET_ITEM(lst, i);
            Py_INCREF(vec[i]);
        }
        self->st = new Set4PyObject(vec);
        Py_SIZE(self) = siz;
    }
    return 0;
}
static PyObject* CppSet_add(CppSet* self, PyObject* args){
    PyObject* x;
    PARSE_ARGS("O", &x);
    bool res = self->st->add(x);
    if(res) Py_SIZE(self)++;
    return Py_BuildValue("O", res ? Py_True : Py_False);
}
static PyObject* CppSet_remove(CppSet* self, PyObject* args){
    PyObject* x;
    PARSE_ARGS("O", &x);
    PyObject* res = self->st->remove(x);
    if(res==NULL) return (PyObject*)NULL;
    Py_SIZE(self)--;
    return Py_BuildValue("O", res);
}
static PyObject* CppSet_search_higher_equal(CppSet* self, PyObject* args){
    PyObject* x;
    PARSE_ARGS("O", &x);
    PyObject* res = self->st->search_higher_equal(x);
    return Py_BuildValue("O", res);
}
static PyObject* CppSet_min(CppSet* self, PyObject* args){
    PyObject* res = self->st->min();
    return Py_BuildValue("O", res);
}
static PyObject* CppSet_max(CppSet* self, PyObject* args){
    PyObject* res = self->st->max();
    return Py_BuildValue("O", res);
}
static PyObject* CppSet_pop_min(CppSet* self, PyObject* args){
    PyObject* res = self->st->pop_min();
    if(res==NULL) return (PyObject*)NULL;
    Py_SIZE(self)--;
    return res;  // 参照カウントを増やさない
}
static PyObject* CppSet_pop_max(CppSet* self, PyObject* args){
    PyObject* res = self->st->pop_max();
    if(res==NULL) return (PyObject*)NULL;
    Py_SIZE(self)--;
    return res;  // 参照カウントを増やさない
}
static Py_ssize_t CppSet_len(CppSet* self){
    return Py_SIZE(self);
}
static PyObject* CppSet_next(CppSet* self, PyObject* args){
    PyObject* res = self->st->iter_next();
    return Py_BuildValue("O", res);
}
static PyObject* CppSet_prev(CppSet* self, PyObject* args){
    PyObject* res = self->st->iter_prev();
    return Py_BuildValue("O", res);
}
static PyObject* CppSet_to_list(CppSet* self, PyObject* args){
    PyObject* res = self->st->to_list();
    return res;
}
static PyObject* CppSet_get(CppSet* self, PyObject* args){
    PyObject* res = self->st->get();
    return Py_BuildValue("O", res);
}
static PyObject* CppSet_erase(CppSet* self, PyObject* args){
    PyObject* res = self->st->erase();
    if(res==NULL) return (PyObject*)NULL;
    Py_SIZE(self)--;
    return Py_BuildValue("O", res);
}
static PyObject* CppSet_copy(CppSet* self, PyObject* args){
    CppSet* st2 = (CppSet*)CppSet_new(&CppSetType, (PyObject*)NULL, (PyObject*)NULL);
    if (st2==NULL) return (PyObject*)NULL;
    st2->st = new Set4PyObject(*self->st);
    Py_SIZE(st2) = Py_SIZE(self);
    return (PyObject*)st2;
}
static PyObject* CppSet_getitem(CppSet* self, Py_ssize_t idx){
    PyObject* res = self->st->getitem((long)idx);
    return Py_BuildValue("O", res);
}
static PyObject* CppSet_pop(CppSet* self, PyObject* args){
    long idx;
    PARSE_ARGS("l", &idx);
    PyObject* res = self->st->pop(idx);
    if(res==NULL) return (PyObject*)NULL;
    Py_SIZE(self)--;
    return Py_BuildValue("O", res);
}
static PyObject* CppSet_index(CppSet* self, PyObject* args){
    PyObject* x;
    PARSE_ARGS("O", &x);
    long res = self->st->index(x);
    return Py_BuildValue("l", res);
}
static int CppSet_contains(CppSet* self, PyObject* x){
    return PyObject_RichCompareBool(self->st->search_higher_equal(x), x, Py_EQ);
}
static int CppSet_bool(CppSet* self){
    return Py_SIZE(self) != 0;
}
static PyObject* CppSet_repr(PyObject* self){
    PyObject *result, *aslist;
    aslist = ((CppSet*)self)->st->to_list();
    result = PyUnicode_FromFormat("CppSet(%R)", aslist);
    Py_ReprLeave(self);
    Py_DECREF(aslist);
    return result;
}

static PyMethodDef CppSet_methods[] = {
    {"add", (PyCFunction)CppSet_add, METH_VARARGS, "Add item"},
    {"remove", (PyCFunction)CppSet_remove, METH_VARARGS, "Remove item"},
    {"search_higher_equal", (PyCFunction)CppSet_search_higher_equal, METH_VARARGS, "Search item"},
    {"min", (PyCFunction)CppSet_min, METH_VARARGS, "Get minimum item"},
    {"max", (PyCFunction)CppSet_max, METH_VARARGS, "Get maximum item"},
    {"pop_min", (PyCFunction)CppSet_pop_min, METH_VARARGS, "Pop minimum item"},
    {"pop_max", (PyCFunction)CppSet_pop_max, METH_VARARGS, "Pop maximum item"},
    {"next", (PyCFunction)CppSet_next, METH_VARARGS, "Get next value"},
    {"prev", (PyCFunction)CppSet_prev, METH_VARARGS, "Get previous value"},
    {"to_list", (PyCFunction)CppSet_to_list, METH_VARARGS, "Make list from set"},
    {"get", (PyCFunction)CppSet_get, METH_VARARGS, "Get item that iterator is pointing at"},
    {"erase", (PyCFunction)CppSet_erase, METH_VARARGS, "Erase item that iterator is pointing at"},
    {"copy", (PyCFunction)CppSet_copy, METH_VARARGS, "Copy set"},
    {"getitem", (PyCFunction)CppSet_getitem, METH_VARARGS, "Get item by index"},
    {"pop", (PyCFunction)CppSet_pop, METH_VARARGS, "Pop item"},
    {"index", (PyCFunction)CppSet_index, METH_VARARGS, "Get index of item"},
    {NULL}  /* Sentinel */
};
static PySequenceMethods CppSet_as_sequence = {
    (lenfunc)CppSet_len,                /* sq_length */
    0,                                  /* sq_concat */
    0,                                  /* sq_repeat */
    (ssizeargfunc)CppSet_getitem,       /* sq_item */
    0,                                  /* sq_slice */
    0,                                  /* sq_ass_item */
    0,                                  /* sq_ass_slice */
    (objobjproc)CppSet_contains,        /* sq_contains */
    0,                                  /* sq_inplace_concat */
    0,                                  /* sq_inplace_repeat */
};
static PyNumberMethods CppSet_as_number = {
    0,                                  /* nb_add */
    0,                                  /* nb_subtract */
    0,                                  /* nb_multiply */
    0,                                  /* nb_remainder */
    0,                                  /* nb_divmod */
    0,                                  /* nb_power */
    0,                                  /* nb_negative */
    0,                                  /* nb_positive */
    0,                                  /* nb_absolute */
    (inquiry)CppSet_bool,               /* nb_bool */
    0,                                  /* nb_invert */
};
PyTypeObject CppSetType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "cppset.CppSet",                    /*tp_name*/
    sizeof(CppSet),                     /*tp_basicsize*/
    0,                                  /*tp_itemsize*/
    (destructor) CppSet_dealloc,        /*tp_dealloc*/
    0,                                  /*tp_print*/
    0,                                  /*tp_getattr*/
    0,                                  /*tp_setattr*/
    0,                                  /*reserved*/
    CppSet_repr,                        /*tp_repr*/
    &CppSet_as_number,                  /*tp_as_number*/
    &CppSet_as_sequence,                /*tp_as_sequence*/
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
    CppSet_methods,                     /*tp_methods*/
    0,                                  /*tp_members*/
    0,                                  /*tp_getset*/
    0,                                  /*tp_base*/
    0,                                  /*tp_dict*/
    0,                                  /*tp_descr_get*/
    0,                                  /*tp_descr_set*/
    0,                                  /*tp_dictoffset*/
    (initproc)CppSet_init,              /*tp_init*/
    0,                                  /*tp_alloc*/
    CppSet_new,                         /*tp_new*/
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

static PyModuleDef cppsetmodule = {
    PyModuleDef_HEAD_INIT,
    "cppset",
    NULL,
    -1,
};

PyMODINIT_FUNC PyInit_cppset(void)
{
    PyObject* m;
    if(PyType_Ready(&CppSetType) < 0) return NULL;

    m = PyModule_Create(&cppsetmodule);
    if(m == NULL) return NULL;

    Py_INCREF(&CppSetType);
    if (PyModule_AddObject(m, "CppSet", (PyObject*) &CppSetType) < 0) {
        Py_DECREF(&CppSetType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
"""
code_setup = r"""
from distutils.core import setup, Extension
module = Extension(
    "cppset",
    sources=["set_wrapper.cpp"],
    extra_compile_args=["-O3", "-march=native", "-std=c++14"]
)
setup(
    name="SetMethod",
    version="0.2.0",
    description="wrapper for C++ set",
    ext_modules=[module]
)
"""

import os
import sys
if sys.argv[-1] == "ONLINE_JUDGE" or os.getcwd() != "/imojudge/sandbox":
    with open("set_wrapper.cpp", "w") as f:
        f.write(code_cppset)
    with open("setup.py", "w") as f:
        f.write(code_setup)
    os.system(f"{sys.executable} setup.py build_ext --inplace")

from cppset import CppSet
