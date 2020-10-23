# 拡張モジュールの方が速いっぽい？

cppset_code = r"""  // 参考: https://atcoder.jp/contests/abc128/submissions/5808742
#include <cstdio>
//#undef __GNUC__
#ifdef __GNUC__
    #include <ext/pb_ds/assoc_container.hpp>
    #include <ext/pb_ds/tree_policy.hpp>
    using namespace std;
    using namespace __gnu_pbds;
    using pb_set = tree<
        long long,
        null_type,
        less<long long>,
        rb_tree_tag,
        tree_order_statistics_node_update
    >;
#else
    #include <set>
    using namespace std;
    using pb_set = set<long long>;
#endif

extern "C" {

void* set_construct(){
    return (void*)(new pb_set);
}

void set_destruct(void* st){
    delete (pb_set*)st;
}

bool set_add(void* st, long long x){
    return ((pb_set*)st)->insert(x).second;
}

void set_remove(void* st, long long x){
    auto it = ((pb_set*)st)->find(x);
    if(it == ((pb_set*)st)->end()){
        //fprintf(stderr, "cppset remove: KeyError\n");
        return;
    }
    ((pb_set*)st)->erase(it);
}

long long set_search_higher_equal(void* st, long long x){
    return *((pb_set*)st)->lower_bound(x);
}

long long set_min(void* st){
    if(((pb_set*)st)->size()==0){
        //fprintf(stderr, "min from an empty set");
        return -1;
    }
    return *((pb_set*)st)->begin();
}

long long set_max(void* st){
    if(((pb_set*)st)->size()==0){
        //fprintf(stderr, "max from an empty set");
        return -1;
    }
    return *prev(((pb_set*)st)->end());
}

long long set_pop_min(void* st){
    if(((pb_set*)st)->size()==0){
        //fprintf(stderr, "pop_min from an empty set");
        return -1;
    }
    auto it = ((pb_set*)st)->begin();
    long long res = *it;
    ((pb_set*)st)->erase(it);
    return res;
}

long long set_pop_max(void* st){
    if(((pb_set*)st)->size()==0){
        //fprintf(stderr, "pop_max from an empty set");
        return -1;
    }
    auto it = prev(((pb_set*)st)->end());
    long long res = *it;
    ((pb_set*)st)->erase(it);
	return res;
}

long long set_len(void* st){
    return ((pb_set*)st)->size();
}

bool set_contains(void* st, long long x){
    return ((pb_set*)st)->find(x) != ((pb_set*)st)->end();
}

long long set_getitem(void* st_, long long idx){
    pb_set* st = (pb_set*)st_;
    long long idx_pos = idx >= 0 ? idx : idx + (long long)st->size();
    if(idx_pos >= (long long)st->size() || idx_pos < 0){
        //fprintf(stderr, "cppset getitem: index out of range\n");
        return -1;
    }
    #ifdef __GNUC__
        auto it = st->find_by_order(idx_pos);
    #else
        auto it = st->begin();
        for(int i=0; i<idx_pos; i++) it++;
    #endif
    return *it;
}

long long set_pop(void* st_, long long idx){
    pb_set* st = (pb_set*)st_;
    long long idx_pos = idx >= 0 ? idx : idx + (long long)st->size();
    if(idx_pos >= (long long)st->size() || idx_pos < 0){
        //fprintf(stderr, "cppset pop: index out of range\n");
        return -1;
    }
    #ifdef __GNUC__
        auto it = st->find_by_order(idx_pos);
    #else
        auto it = st->begin();
        for(int i=0; i<idx_pos; i++) it++;
    #endif
    long long res = *it;
    st->erase(it);
    return res;
}

long long set_index(void* st, long long x){
    #ifdef __GNUC__
        return ((pb_set*)st)->order_of_key(x);
    #else
        long long res = 0;
        auto it = ((pb_set*)st)->begin();
        while(it != ((pb_set*)st)->end() && *it < x) it++, res++;
        return res;
    #endif
}


}  // extern "C"
"""

import os
import sys
from functools import partial
import distutils.ccompiler
from ctypes import CDLL, CFUNCTYPE, c_bool, c_longlong, c_void_p

if sys.argv[-1] == "ONLINE_JUDGE" or os.getcwd() != "/imojudge/sandbox":
    with open("cppset.cpp", "w") as f:
        f.write(cppset_code)
    if os.name == "nt":
        os.system(r'"C:\Program Files\mingw-w64\x86_64-8.1.0-posix-seh-rt_v6-rev0\mingw64\bin\g++" -fPIC -shared -std=c++14 -O3 cppset.cpp -o cppset.dll')
        # compiler = distutils.ccompiler.new_compiler()
        # compiler.compile(["cppset.cpp"], extra_postargs=["/LD"])
        # link_args = ["/DLL"]
        # compiler.link_shared_lib(["cppset.obj"], "cppset", extra_postargs=link_args)
    else:
        os.system(f"g++ -fPIC -shared -std=c++14 -O3 cppset.cpp -o cppset.so")
if os.name == "nt":
    lib = CDLL(f"{os.getcwd()}/cppset.dll")
else:
    lib = CDLL(f"{os.getcwd()}/cppset.so")


class CppSetInt:
    def __init__(self):
        self.ptr = CFUNCTYPE(c_void_p)(("set_construct", lib))()
        self.add = partial(CFUNCTYPE(c_bool, c_void_p, c_longlong)(("set_add", lib)), self.ptr)
        self.remove = partial(CFUNCTYPE(None, c_void_p, c_longlong)(("set_remove", lib)), self.ptr)
        self.search_higher_equal = partial(
            CFUNCTYPE(c_longlong, c_void_p, c_longlong)(("set_search_higher_equal", lib)), self.ptr)
        self.min = partial(CFUNCTYPE(c_longlong, c_void_p)(("set_min", lib)), self.ptr)
        self.max = partial(CFUNCTYPE(c_longlong, c_void_p)(("set_max", lib)), self.ptr)
        self.pop_min = partial(CFUNCTYPE(c_longlong, c_void_p)(("set_pop_min", lib)), self.ptr)
        self.pop_max = partial(CFUNCTYPE(c_longlong, c_void_p)(("set_pop_max", lib)), self.ptr)
        self.__len__ = partial(CFUNCTYPE(c_longlong, c_void_p)(("set_len", lib)), self.ptr)
        self.contains = partial(CFUNCTYPE(c_bool, c_void_p, c_longlong)(("set_contains", lib)), self.ptr)
        self.__getitem__ = partial(CFUNCTYPE(c_longlong, c_void_p, c_longlong)(("set_getitem", lib)), self.ptr)
        self.pop = partial(CFUNCTYPE(c_longlong, c_void_p, c_longlong)(("set_pop", lib)), self.ptr)
        self.index = partial(CFUNCTYPE(c_longlong, c_void_p, c_longlong)(("set_index", lib)), self.ptr)

    def __len__(self):
        return self.__len__()

    def __contains__(self):
        return self.__contains__()

