code_cppset = r"""
#include <Python.h>
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
#define GET_POINTER(stCapsule) (Set4PyObject*)PyCapsule_GetPointer(stCapsule, "Set4PyObjectPtr")

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
        if(it == st.end()) PyErr_SetString(PyExc_KeyError, "erase end"), (PyObject*)NULL;
        PyObject* res = *it;
        it = st.erase(it);
        if(it == st.end()) return Py_None;
        return *it;
    }
    PyObject* getitem(const long& idx){
        long idx_pos = idx >= 0 ? idx : idx + (long)st.size();
        if(idx_pos >= st.size() || idx_pos < 0)
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
        if(idx_pos >= st.size() || idx_pos < 0)
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

void Set4PyObject_free(PyObject *obj){
    Set4PyObject* const p = (Set4PyObject*)PyCapsule_GetPointer(obj, "Set4PyObjectPtr");
    delete p;
}
PyObject* Set4PyObject_construct(PyObject* self, PyObject* args){
    Set4PyObject* st = new Set4PyObject();
    return PyCapsule_New((void*)st, "Set4PyObjectPtr", Set4PyObject_free);
}
PyObject* Set4PyObject_construct_from_list(PyObject* self, PyObject* args){
    PyObject* lst;
    PARSE_ARGS("O", &lst);
    int siz;
    if(PyList_Check(lst)) siz = (int)PyList_GET_SIZE(lst);
    else if(PyTuple_Check(lst)) siz = (int)PyTuple_GET_SIZE(lst);
    else PyErr_SetString(PyExc_TypeError, "got neither list nor tuple");
    vector<PyObject*> vec(siz);
    for(int i=0; i<siz; i++){
        vec[i] = PyList_Check(lst) ? PyList_GET_ITEM(lst, i) : PyTuple_GET_ITEM(lst, i);
        Py_INCREF(vec[i]);
    }
    Set4PyObject* st = new Set4PyObject(vec);
    return PyCapsule_New((void*)st, "Set4PyObjectPtr", Set4PyObject_free);
}
PyObject* Set4PyObject_add(PyObject* self, PyObject* args){
    PyObject *stCapsule, *x;
    PARSE_ARGS("OO", &stCapsule, &x);
    Set4PyObject* st = GET_POINTER(stCapsule);
    bool res = st->add(x);
    return Py_BuildValue("O", res ? Py_True : Py_False);
}
PyObject* Set4PyObject_remove(PyObject* self, PyObject* args){
    PyObject *stCapsule, *x;
    PARSE_ARGS("OO", &stCapsule, &x);
    Set4PyObject* st = GET_POINTER(stCapsule);
    PyObject* res = st->remove(x);
    return Py_BuildValue("O", res);
}
PyObject* Set4PyObject_search_higher_equal(PyObject* self, PyObject* args){
    PyObject *stCapsule, *x;
    PARSE_ARGS("OO", &stCapsule, &x);
    Set4PyObject* st = GET_POINTER(stCapsule);
    PyObject* res = st->search_higher_equal(x);
    return Py_BuildValue("O", res);
}
PyObject* Set4PyObject_min(PyObject* self, PyObject* args){
    PyObject *stCapsule;
    PARSE_ARGS("O", &stCapsule);
    Set4PyObject* st = GET_POINTER(stCapsule);
    PyObject* res = st->min();
    return Py_BuildValue("O", res);
}
PyObject* Set4PyObject_max(PyObject* self, PyObject* args){
    PyObject *stCapsule;
    PARSE_ARGS("O", &stCapsule);
    Set4PyObject* st = GET_POINTER(stCapsule);
    PyObject* res = st->max();
    return Py_BuildValue("O", res);
}
PyObject* Set4PyObject_pop_min(PyObject* self, PyObject* args){
    PyObject *stCapsule;
    PARSE_ARGS("O", &stCapsule);
    Set4PyObject* st = GET_POINTER(stCapsule);
    PyObject* res = st->pop_min();
    return res;  // 参照カウントを増やさない
}
PyObject* Set4PyObject_pop_max(PyObject* self, PyObject* args){
    PyObject *stCapsule;
    PARSE_ARGS("O", &stCapsule);
    Set4PyObject* st = GET_POINTER(stCapsule);
    PyObject* res = st->pop_max();
    return res;  // 参照カウントを増やさない
}
PyObject* Set4PyObject_len(PyObject* self, PyObject* args){
    PyObject *stCapsule;
    PARSE_ARGS("O", &stCapsule);
    Set4PyObject* st = GET_POINTER(stCapsule);
    size_t res = st->len();
    return Py_BuildValue("l", (long)res);
}
PyObject* Set4PyObject_next(PyObject* self, PyObject* args){
    PyObject *stCapsule;
    PARSE_ARGS("O", &stCapsule);
    Set4PyObject* st = GET_POINTER(stCapsule);
    PyObject* res = st->iter_next();
    return Py_BuildValue("O", res);
}
PyObject* Set4PyObject_prev(PyObject* self, PyObject* args){
    PyObject *stCapsule;
    PARSE_ARGS("O", &stCapsule);
    Set4PyObject* st = GET_POINTER(stCapsule);
    PyObject* res = st->iter_prev();
    return Py_BuildValue("O", res);
}
PyObject* Set4PyObject_to_list(PyObject* self, PyObject* args){
    PyObject *stCapsule;
    PARSE_ARGS("O", &stCapsule);
    Set4PyObject* st = GET_POINTER(stCapsule);
    PyObject* res = st->to_list();
    return res;  // 参照カウントを増やさない
}
PyObject* Set4PyObject_get(PyObject* self, PyObject* args){
    PyObject *stCapsule;
    PARSE_ARGS("O", &stCapsule);
    Set4PyObject* st = GET_POINTER(stCapsule);
    PyObject* res = st->get();
    return Py_BuildValue("O", res);
}
PyObject* Set4PyObject_erase(PyObject* self, PyObject* args){
    PyObject *stCapsule;
    PARSE_ARGS("O", &stCapsule);
    Set4PyObject* st = GET_POINTER(stCapsule);
    PyObject* res = st->erase();
    return Py_BuildValue("O", res);
}
PyObject* Set4PyObject_copy(PyObject* self, PyObject* args){
    PyObject *stCapsule;
    PARSE_ARGS("O", &stCapsule);
    Set4PyObject* st = GET_POINTER(stCapsule);
    Set4PyObject* st2 = new Set4PyObject(*st);
    return PyCapsule_New((void*)st2, "Set4PyObjectPtr", Set4PyObject_free);
}
PyObject* Set4PyObject_getitem(PyObject* self, PyObject* args){
    PyObject *stCapsule;
    long idx;
    PARSE_ARGS("Ol", &stCapsule, &idx);
    Set4PyObject* st = GET_POINTER(stCapsule);
    PyObject* res = st->getitem(idx);
    return Py_BuildValue("O", res);
}
PyObject* Set4PyObject_pop(PyObject* self, PyObject* args){
    PyObject *stCapsule;
    long idx;
    PARSE_ARGS("Ol", &stCapsule, &idx);
    Set4PyObject* st = GET_POINTER(stCapsule);
    PyObject* res = st->pop(idx);
    return res;  // 参照カウントを増やさない
}
PyObject* Set4PyObject_index(PyObject* self, PyObject* args){
    PyObject *stCapsule, *x;
    PARSE_ARGS("OO", &stCapsule, &x);
    Set4PyObject* st = GET_POINTER(stCapsule);
    long res = st->index(x);
    return Py_BuildValue("l", res);
}
static PyMethodDef SetMethods[] = {
    {"construct", Set4PyObject_construct, METH_VARARGS, "Create set object"},
    {"construct_from_list", Set4PyObject_construct_from_list, METH_VARARGS, "Create set object from list"},
    {"add", Set4PyObject_add, METH_VARARGS, "Add item"},
    {"remove", Set4PyObject_remove, METH_VARARGS, "Remove item"},
    {"search_higher_equal", Set4PyObject_search_higher_equal, METH_VARARGS, "Search item"},
    {"min", Set4PyObject_min, METH_VARARGS, "Get minimum item"},
    {"max", Set4PyObject_max, METH_VARARGS, "Get maximum item"},
    {"pop_min", Set4PyObject_pop_min, METH_VARARGS, "Pop minimum item"},
    {"pop_max", Set4PyObject_pop_max, METH_VARARGS, "Pop maximum item"},
    {"len", Set4PyObject_len, METH_VARARGS, "Get size"},
    {"next", Set4PyObject_next, METH_VARARGS, "Get next value"},
    {"prev", Set4PyObject_prev, METH_VARARGS, "Get previous value"},
    {"to_list", Set4PyObject_to_list, METH_VARARGS, "Make list from set"},
    {"get", Set4PyObject_get, METH_VARARGS, "Get item that iterator is pointing at"},
    {"erase", Set4PyObject_erase, METH_VARARGS, "Erase item that iterator is pointing at"},
    {"copy", Set4PyObject_copy, METH_VARARGS, "Copy set"},
    {"getitem", Set4PyObject_getitem, METH_VARARGS, "Get item by index"},
    {"pop", Set4PyObject_pop, METH_VARARGS, "Pop item"},
    {"index", Set4PyObject_index, METH_VARARGS, "Get index of item"},
    {NULL, NULL, 0, NULL} 
};

static struct PyModuleDef setmodule = {
    PyModuleDef_HEAD_INIT,
    "cppset", 
    NULL, 
    -1, 
    SetMethods,
};

PyMODINIT_FUNC PyInit_cppset(void){
    return PyModule_Create(&setmodule);
}
"""
code_setup = r"""
from distutils.core import setup, Extension
module = Extension(
    "cppset",
    sources=["set_wrapper.cpp"],
    extra_compile_args=["-O3", "-march=native"]
)
setup(
    name="SetMethod",
    version="0.1.0",
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


import cppset
class CppSet:
    def __init__(self, lst=None, capsule=None):
        if lst:
            self.st = cppset.construct_from_list(lst)
        elif capsule:
            self.st = capsule
        else:
            self.st = cppset.construct()

    def add(self, val):
        return cppset.add(self.st, val)

    def remove(self, val):
        return cppset.remove(self.st, val)

    def search_higher_equal(self, val):
        return cppset.search_higher_equal(self.st, val)

    def min(self):
        return cppset.min(self.st)

    def max(self):
        return cppset.max(self.st)

    def pop_min(self):
        return cppset.pop_min(self.st)

    def pop_max(self):
        return cppset.pop_max(self.st)

    def __len__(self):
        return cppset.len(self.st)

    def __getitem__(self, idx):
        # g++ 拡張
        return cppset.getitem(self.st, idx)

    def pop(self, idx):
        # g++ 拡張
        return cppset.pop(self.st, idx)

    def index(self, val):
        # g++ 拡張  # イテレータは変化しない
        return cppset.index(self.st, val)

    def __contains__(self, val):
        return self.search_higher_equal(val) == val

    def next(self):
        return cppset.next(self.st)

    def prev(self):
        return cppset.prev(self.st)

    def to_list(self):
        return cppset.to_list(self.st)

    def get(self):
        return cppset.get(self.st)

    def erase(self):
        return cppset.erase(self.st)

    def copy(self):
        return CppSet(capsule=cppset.copy(self.st))

    def __repr__(self):
        return f"CppSet({self.to_list().__repr__()})"
