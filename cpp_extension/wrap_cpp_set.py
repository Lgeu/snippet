code_cppset = r"""
#include <Python.h>
#include <vector>
#ifdef __GNUC__
    #include <ext/pb_ds/assoc_container.hpp>
    #include <ext/pb_ds/tree_policy.hpp>
    using namespace std;
    using namespace __gnu_pbds;
    const static auto comp_pyobj = [](PyObject* const & lhs, PyObject* const & rhs){
        return (bool)PyObject_RichCompareBool(lhs, rhs, Py_LT);
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

// 参考1: http://www.speedupcode.com/c-class-in-python3/
// 参考2: https://qiita.com/junkoda/items/2b1eda7569186809ca14

struct Set4PyObject{
    pb_set st;
    Set4PyObject() : st(comp_pyobj) {}
    Set4PyObject(vector<PyObject*>& vec) : st(vec.begin(), vec.end(), comp_pyobj) {}
    ~Set4PyObject(){
        for(PyObject* const & p : st){
            Py_DECREF(p);
        }
    }
    bool add(PyObject* x){
        if(st.insert(x).second){
            Py_INCREF(x);
            return true;
        }else{
            return false;
        }
    }
    void remove(PyObject* x){
        auto it = st.find(x);
        Py_DECREF(*it);
        st.erase(it);
    }
    PyObject* search_higher_equal(PyObject* x) const {
        return *st.lower_bound(x);
    }
    PyObject* min() const {
        return *st.begin();
    }
    PyObject* max() const {
        return *st.rbegin();
    }
    PyObject* pop_min(){
        auto it = st.begin();
        PyObject* res = *it;
        st.erase(it);
        return res;
    }
    PyObject* pop_max(){
        auto it = prev(st.end());
        PyObject* res = *it;
        st.erase(it);
        return res;
    }
    long len() const {
        return st.size();
    }
    #ifdef __GNUC__
        /// g++ 拡張 ///
        PyObject* getitem(const long& idx) const {
            return *st.find_by_order(idx);
        }
        PyObject* pop(const long& idx){
            // idx 番目の要素が存在しないとエラー
            auto it = st.find_by_order(idx);
            st.erase(it);
            return *it;
        }
        long index(PyObject* x){
            return st.order_of_key(x);
        }
    #endif
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
    if(PyList_Check(lst)) siz = PyList_GET_SIZE(lst);
    else if(PyTuple_Check(lst)) siz = PyTuple_GET_SIZE(lst);
    else return NULL;
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
    st->add(x);
    return Py_BuildValue("");
}
PyObject* Set4PyObject_remove(PyObject* self, PyObject* args){
    PyObject *stCapsule, *x;
    PARSE_ARGS("OO", &stCapsule, &x);
    Set4PyObject* st = GET_POINTER(stCapsule);
    st->remove(x);
    return Py_BuildValue("");
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
    long res = st->len();
    return Py_BuildValue("l", res);
}
#ifdef __GNUC__
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
#endif
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
    #ifdef __GNUC__
        {"getitem", Set4PyObject_getitem, METH_VARARGS, "Get item by index"},
        {"pop", Set4PyObject_pop, METH_VARARGS, "Pop item"},
        {"index", Set4PyObject_index, METH_VARARGS, "Get index of item"},
    #endif
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
    version="0.0.5",
    description="wrapper for C++ set",
    ext_modules=[module]
)
"""

import os
import sys
if sys.argv[-1] == "ONLINE_JUDGE" or os.name == "nt":
    with open("set_wrapper.cpp", "w") as f:
        f.write(code_cppset)
    with open("setup.py", "w") as f:
        f.write(code_setup)
    os.system(f"{sys.executable} setup.py build_ext --inplace")

import cppset
class CppSet:
    def __init__(self, lst=None):
        if lst:
            self.st = cppset.construct_from_list(lst)
        else:
            self.st = cppset.construct()

    def add(self, val):
        cppset.add(self.st, val)

    def remove(self, val):
        cppset.remove(self.st, val)

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
        # g++ のみ
        siz = self.__len__()
        if idx >= siz:
            raise IndexError
        elif idx < 0:
            idx += siz
            if idx < 0:
                raise IndexError
        return cppset.getitem(self.st, idx)

    def pop(self, idx):
        # g++ のみ
        siz = self.__len__()
        if idx >= siz:
            raise IndexError
        elif idx < 0:
            idx += siz
            if idx < 0:
                raise IndexError
        return cppset.pop(self.st, idx)

    def index(self, val):
        # g++ のみ
        return cppset.index(self.st, val)

    def __contains__(self, val):
        return self.search_higher_equal(val) == val