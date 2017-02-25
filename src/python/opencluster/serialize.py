import sys, types
from cStringIO import StringIO
import marshal, cPickle
import itertools
from pickle import Pickler, whichmodule
import logging
logger = logging.getLogger(__name__)

class MyPickler(Pickler):
    dispatch = Pickler.dispatch.copy()

    @classmethod
    def register(cls, type, reduce):
        def dispatcher(self, obj):
            rv = reduce(obj)
            if isinstance(rv, str):
                self.save_global(obj, rv)
            else:
                self.save_reduce(obj=obj, *rv)
        cls.dispatch[type] = dispatcher

def dumps(o):
    io = StringIO()
    MyPickler(io, -1).dump(o)
    return io.getvalue()

def loads(s):
    return cPickle.loads(s)

dump_func = dumps
load_func = loads

def reduce_module(mod):
    return load_module, (mod.__name__, )

def load_module(name):
    __import__(name)
    return sys.modules[name]

MyPickler.register(types.ModuleType, reduce_module)

class RecursiveFunctionPlaceholder(object):
    """
    Placeholder for a recursive reference to the current function,
    to avoid infinite recursion when serializing recursive functions.
    """
    def __eq__(self, other):
        return isinstance(other, RecursiveFunctionPlaceholder)

RECURSIVE_FUNCTION_PLACEHOLDER = RecursiveFunctionPlaceholder()

def marshalable(o):
    if o is None: return True
    t = type(o)
    if t in (str, unicode, bool, int, long, float, complex):
        return True
    if t in (tuple, list, set):
        for i in itertools.islice(o, 100):
            if not marshalable(i):
                return False
        return True
    if t == dict:
        for k,v in itertools.islice(o.iteritems(), 100):
            if not marshalable(k) or not marshalable(v):
                return False
        return True
    return False

OBJECT_SIZE_LIMIT = 100 << 10


def get_co_names(code):
    co_names = code.co_names
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            co_names += get_co_names(const)

    return co_names


def make_cell(value):
    return (lambda: value).func_closure[0]

def reconstruct_closure(values):
    return tuple([make_cell(v) for v in values])

def get_global_function(module, name):
    __import__(module)
    mod = sys.modules[module]
    return getattr(mod, name)


classes_dumping = set()
internal_fields = {
    '__weakref__': False,
    '__dict__': False,
    '__doc__': True
}
def dump_local_class(cls):
    name = cls.__name__
    if cls in classes_dumping:
        return dumps(name)

    classes_dumping.add(cls)
    internal = {}
    external = {}
    for k in cls.__dict__:
        if k not in internal_fields:
            v = getattr(cls, k)
            if isinstance(v, property):
                k = ('property', k)
                v = (v.fget, v.fset, v.fdel, v.__doc__)

            if isinstance(v, types.FunctionType):
                k = ('staticmethod', k)

            external[k] = v

        elif internal_fields[k]:
            internal[k] = getattr(cls, k)

    result = dumps((cls.__name__, cls.__bases__, internal, dumps(external)))
    if cls in classes_dumping:
        classes_dumping.remove(cls)

    return result

classes_loaded = {}
def load_local_class(bytes):
    t = loads(bytes)
    if not isinstance(t, tuple):
        return classes_loaded[t]

    name, bases, internal, external = t
    if name in classes_loaded:
        return classes_loaded[name]

    cls = type(name, bases, internal)
    classes_loaded[name] = cls
    for k, v in loads(external).items():
        if isinstance(k, tuple):
            t, k = k
            if t == 'property':
                fget, fset, fdel, doc = v
                v = property(fget, fset, fdel, doc)

            if t == 'staticmethod':
                v = staticmethod(v)

        setattr(cls, k, v)

    return cls

def reduce_class(obj):
    name = obj.__name__
    module = getattr(obj, "__module__", None)
    if module == '__main__' and name not in ('MyPickler', 'RecursiveFunctionPlaceholder'):
        result = load_local_class, (dump_local_class(obj),)
        return result

    return name

CLS_TYPES = [types.TypeType, types.ClassType]
def dump_method(method):
    obj = method.im_self
    cls = method.im_class
    func = method.im_func
    if cls in CLS_TYPES:
        cls_name = CLS_TYPES.index(cls)
    else:
        cls_name = cls.__name__

    return dumps((obj, cls_name, func))

def load_method(bytes):
    obj, cls_name, func = loads(bytes) # cls referred in func.func_globals
    if isinstance(cls_name, int):
        cls = CLS_TYPES[cls_name]
    else:
        cls = classes_loaded[cls_name]

    return types.MethodType(func, obj, cls)

def reduce_method(method):
    module = method.im_func.__module__
    return load_method, (dump_method(method), )

MyPickler.register(types.ClassType, reduce_class)
MyPickler.register(types.TypeType, reduce_class)
MyPickler.register(types.MethodType, reduce_method)

if __name__ == "__main__":
    assert marshalable(None)
    assert marshalable("")
    assert marshalable(u"")
    assert not marshalable(buffer(""))
    assert marshalable(0)
    assert marshalable(0L)
    assert marshalable(0.0)
    assert marshalable(True)
    assert marshalable(complex(1,1))
    assert marshalable((1,1))
    assert marshalable([1,1])
    assert marshalable(set([1,1]))
    assert marshalable({1:None})

    some_global = 'some global'
    def glob_func(s):
        return "glob:" + s
    def get_closure(x):
        glob_func(some_global)
        last = " last"
        def foo(y): return "foo: " + y
        def the_closure(a, b=1):
            marshal.dumps(a)
            return (a * x + int(b), glob_func(foo(some_global)+last))
        return the_closure

    f = get_closure(10)
    ff = loads(dumps(f))
    #print globals()
    print f(2)
    print ff(2)
    glob_func = loads(dumps(glob_func))
    get_closure = loads(dumps(get_closure))

    # Test recursive functions
    def fib(n): return n if n <= 1 else fib(n-1) + fib(n-2)
    assert fib(8) == loads(dumps(fib))(8)

    class Foo1:
        def foo(self):
            return 1234

    class Foo2(object):
        def foo(self):
            return 5678

    class Foo3(Foo2):
        x = 1111

        def foo(self):
            return super(Foo3, self).foo() + Foo3.x

    class Foo4(object):
        @classmethod
        def x(cls):
            return 1

        @property
        def y(self):
            return 2

        @staticmethod
        def z():
            return 3

    df1 = dumps(Foo1)
    df2 = dumps(Foo2)
    df3 = dumps(Foo3)
    df4 = dumps(Foo4)

    del Foo1
    del Foo2
    del Foo3
    del Foo4

    Foo1 = loads(df1)
    Foo2 = loads(df2)
    Foo3 = loads(df3)
    Foo4 = loads(df4)

    f1 = Foo1()
    f2 = Foo2()
    f3 = Foo3()
    f4 = Foo4()

    assert f1.foo() == 1234
    assert f2.foo() == 5678
    assert f3.foo() == 5678 + 1111
    assert Foo4.x() == 1

    f = loads(dumps(lambda:(some_global for i in xrange(1))))
    print list(f())
    assert list(f()) == [some_global]
