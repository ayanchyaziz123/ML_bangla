export const pythonEn = [
  {
    slug: 'python-interview-1-50',
    title: 'Python Interview Questions (Beginner → Intermediate) — 50 Q&A with Examples',
    description: '50 must-know Python interview questions covering core language basics, data structures, OOP, decorators, generators, and memory management — each with a code example.',
    category: 'Python Interview Questions',
    readTime: 30,
    content: `
<h3>Core Python Basics (Q1–Q15)</h3>

<p><strong>Q1. What's the difference between a list and a tuple?</strong><br/>
Lists are mutable (you can add/remove/change elements). Tuples are immutable, slightly faster, and can be used as dict keys or set members (if all elements are hashable) — lists cannot.
<pre><code>my_list = [1, 2, 3]
my_list.append(4)          # works fine

my_tuple = (1, 2, 3)
my_tuple[0] = 99            # TypeError: 'tuple' object does not support item assignment</code></pre></p>

<p><strong>Q2. What is the difference between == and is?</strong><br/>
== compares values (calls <code>__eq__</code>). <code>is</code> compares object identity (same memory address). Two equal-value objects can still be different objects.
<pre><code>a = [1, 2, 3]
b = [1, 2, 3]
print(a == b)   # True  (same values)
print(a is b)   # False (different objects)
print(a is a)   # True  (same object)</code></pre></p>

<p><strong>Q3. What are mutable and immutable types in Python?</strong><br/>
Mutable: list, dict, set (can change in place). Immutable: int, float, str, tuple, frozenset, bool — any "change" actually creates a new object.
<pre><code>s = "hello"
s[0] = 'H'          # TypeError — strings are immutable
s = 'H' + s[1:]     # this creates a brand new string object</code></pre></p>

<p><strong>Q4. Explain Python's dynamic typing.</strong><br/>
Variables aren't bound to a fixed type — the same name can be reassigned to any type at runtime, and types are checked at runtime, not compile time.
<pre><code>x = 5
x = "now a string"
x = [1, 2, 3]       # no error — x's type simply changes each time</code></pre></p>

<p><strong>Q5. What is the difference between deep copy and shallow copy?</strong><br/>
A shallow copy copies the outer container, but nested objects are still shared references. A deep copy recursively copies everything, so nested objects become fully independent.
<pre><code>import copy
a = [[1, 2], [3, 4]]
b = copy.copy(a)
b[0][0] = 99         # a[0][0] also becomes 99 — inner list is shared!

c = copy.deepcopy(a)
c[0][0] = 100        # a is completely unaffected</code></pre></p>

<p><strong>Q6. What are *args and **kwargs?</strong><br/>
<code>*args</code> collects extra positional arguments into a tuple. <code>**kwargs</code> collects extra keyword arguments into a dict — used for functions that need to accept a variable number of arguments.
<pre><code>def f(*args, **kwargs):
    print(args, kwargs)

f(1, 2, a=3, b=4)   # (1, 2) {'a': 3, 'b': 4}</code></pre></p>

<p><strong>Q7. Is Python "pass by value" or "pass by reference"?</strong><br/>
Neither exactly — Python passes object references by value ("pass by object reference"). For immutable arguments the function can't change the caller's variable. For mutable arguments the function can modify the object in place, but reassigning the parameter name doesn't affect the caller.
<pre><code>def modify(lst): lst.append(4)
def reassign(lst): lst = [100]

a = [1, 2, 3]
modify(a)     # a becomes [1, 2, 3, 4] — in-place mutation is visible
reassign(a)   # a stays [1, 2, 3, 4] — only the local name was rebound</code></pre></p>

<p><strong>Q8. What is a lambda function?</strong><br/>
An anonymous, single-expression function defined inline — commonly used as a short callback for sorting, filtering, or mapping.
<pre><code>pairs = [(1, 'b'), (2, 'a')]
sorted(pairs, key=lambda x: x[1])   # sorts by the second element</code></pre></p>

<p><strong>Q9. Explain list comprehension vs generator expression.</strong><br/>
A list comprehension <code>[x for x in ...]</code> builds the entire list in memory immediately. A generator expression <code>(x for x in ...)</code> produces items lazily, one at a time — using far less memory for large sequences.
<pre><code>squares_list = [x**2 for x in range(1000000)]   # builds the full list in memory
squares_gen  = (x**2 for x in range(1000000))   # lazy — near-zero memory until consumed</code></pre></p>

<p><strong>Q10. What is the difference between remove(), pop(), and del for lists?</strong><br/>
<code>remove(value)</code> removes the first matching value. <code>pop(index)</code> removes and returns the item at that index (default: last). <code>del list[index]</code> removes by index without returning the value; <code>del</code> can also delete the whole variable.
<pre><code>lst = [1, 2, 3]
lst.remove(2)   # [1, 3]
lst.pop()       # returns 3, lst is now [1]
del lst[0]      # removes index 0</code></pre></p>

<p><strong>Q11. What are Python's built-in data types?</strong><br/>
Numeric (int, float, complex), sequence (str, list, tuple, range), mapping (dict), set types (set, frozenset), boolean (bool), and the None type.
<pre><code>type(5), type(5.0), type("a"), type([1]), type({})
# int, float, str, list, dict</code></pre></p>

<p><strong>Q12. What is string interning in Python?</strong><br/>
CPython automatically caches (interns) small strings and identifiers, so identical string literals may point to the same object in memory as an optimization — this affects <code>is</code> comparisons on strings but shouldn't be relied on for correctness.
<pre><code>a = "hello"
b = "hello"
print(a is b)   # often True due to interning — but not guaranteed for all strings</code></pre></p>

<p><strong>Q13. What does the <code>pass</code> statement do?</strong><br/>
A no-op placeholder statement, used where Python's syntax requires a statement but no action is needed yet — like an empty function body or a stub class.
<pre><code>def todo_later():
    pass   # implement this later</code></pre></p>

<p><strong>Q14. What is the difference between <code>break</code>, <code>continue</code>, and <code>pass</code>?</strong><br/>
<code>break</code> exits the loop entirely. <code>continue</code> skips to the next iteration. <code>pass</code> does nothing — it just satisfies syntax.
<pre><code>for i in range(5):
    if i == 2: continue   # skip 2, keep looping
    if i == 4: break       # stop at 4
    print(i)                # prints 0, 1, 3</code></pre></p>

<p><strong>Q15. How does Python manage memory?</strong><br/>
Through reference counting (an object is freed the moment its reference count hits zero) combined with a generational garbage collector that detects and cleans up reference cycles reference counting alone can't catch.
<pre><code>import gc
gc.collect()   # forces a collection cycle to clean up unreachable reference cycles</code></pre></p>

<h3>Data Structures &amp; Functions (Q16–Q30)</h3>

<p><strong>Q16. What's the difference between a set and a list?</strong><br/>
Sets store unique, unordered elements with O(1) average membership testing. Lists allow duplicates, preserve insertion order, and membership testing is O(n).
<pre><code>list(set([1, 2, 2, 3, 3, 3]))   # [1, 2, 3] — duplicates removed</code></pre></p>

<p><strong>Q17. What is a dictionary comprehension?</strong><br/>
A concise <code>{k: v for ...}</code> syntax to build a dict from an iterable — analogous to a list comprehension.
<pre><code>{x: x**2 for x in range(5)}
# {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}</code></pre></p>

<p><strong>Q18. Explain Python's <code>*</code> unpacking operator for iterables.</strong><br/>
<code>*</code> unpacks an iterable's elements into individual items — used in function calls, assignments, and merging collections.
<pre><code>a, *rest = [1, 2, 3, 4]     # a = 1, rest = [2, 3, 4]
merged = [*[1, 2], *[3, 4]]  # [1, 2, 3, 4]</code></pre></p>

<p><strong>Q19. What are decorators in Python?</strong><br/>
A decorator is a function that wraps another function to extend its behavior without modifying its source code — implemented using the <code>@decorator</code> syntax.
<pre><code>def timer(func):
    def wrapper(*args, **kwargs):
        import time; start = time.time()
        result = func(*args, **kwargs)
        print(time.time() - start)
        return result
    return wrapper

@timer
def slow_add(a, b):
    return a + b</code></pre></p>

<p><strong>Q20. What is a closure?</strong><br/>
A closure is a nested function that remembers and can access variables from its enclosing function's scope, even after the outer function has returned.
<pre><code>def make_multiplier(n):
    def multiply(x):
        return x * n
    return multiply

times3 = make_multiplier(3)
times3(5)   # 15 — remembers n=3</code></pre></p>

<p><strong>Q21. What's the difference between a function and a generator function?</strong><br/>
A regular function returns a single value with <code>return</code> and its execution ends there. A generator function uses <code>yield</code>, pausing and resuming execution to lazily produce a sequence of values one at a time.
<pre><code>def gen():
    yield 1
    yield 2
    yield 3

for val in gen():
    print(val)   # 1, 2, 3 — one at a time, not all in memory</code></pre></p>

<p><strong>Q22. What is the difference between an iterable and an iterator?</strong><br/>
An iterable is any object you can loop over (implements <code>__iter__</code>), like a list. An iterator is what <code>iter()</code> returns from an iterable — it implements <code>__next__</code> and tracks position, raising <code>StopIteration</code> when exhausted.
<pre><code>lst = [1, 2, 3]   # iterable
it = iter(lst)     # iterator
next(it)   # 1
next(it)   # 2</code></pre></p>

<p><strong>Q23. Explain map(), filter(), and reduce().</strong><br/>
<code>map(func, iterable)</code> applies func to every element. <code>filter(func, iterable)</code> keeps only elements where func returns True. <code>reduce(func, iterable)</code> (from functools) cumulatively combines elements into a single value.
<pre><code>from functools import reduce
nums = [1, 2, 3, 4]
list(map(lambda x: x * 2, nums))        # [2, 4, 6, 8]
list(filter(lambda x: x % 2 == 0, nums)) # [2, 4]
reduce(lambda a, b: a + b, nums)         # 10</code></pre></p>

<p><strong>Q24. What is exception handling? Explain try/except/else/finally.</strong><br/>
<code>try</code> holds code that might raise an exception. <code>except</code> catches specific exception types. <code>else</code> runs only if no exception occurred. <code>finally</code> always runs — used for cleanup (closing files, releasing resources).
<pre><code>try:
    result = 10 / 0
except ZeroDivisionError as e:
    print("Cannot divide by zero:", e)
else:
    print("No error occurred")
finally:
    print("This always runs")</code></pre></p>

<p><strong>Q25. What is the difference between Exception and BaseException?</strong><br/>
<code>BaseException</code> is the root of the exception hierarchy, including <code>SystemExit</code>, <code>KeyboardInterrupt</code>, and <code>GeneratorExit</code>. <code>Exception</code> is a subclass covering normal errors your code should typically catch — catching bare <code>Exception</code> (not <code>BaseException</code>) is best practice so you don't accidentally swallow Ctrl+C or a system exit.
<pre><code>try:
    risky_operation()
except Exception as e:   # catches ValueError, TypeError, etc. — not KeyboardInterrupt
    print(e)</code></pre></p>

<p><strong>Q26. How do you create a custom exception?</strong><br/>
Subclass the built-in <code>Exception</code> class (or a more specific one), optionally overriding <code>__init__</code> to add custom attributes.
<pre><code>class InsufficientFundsError(Exception):
    def __init__(self, balance, amount):
        super().__init__(f"Cannot withdraw {amount}, balance is {balance}")
        self.balance = balance

raise InsufficientFundsError(100, 500)</code></pre></p>

<p><strong>Q27. What is a context manager and the <code>with</code> statement?</strong><br/>
A context manager guarantees setup/teardown code runs around a block, even if an exception occurs — implemented via <code>__enter__</code>/<code>__exit__</code>. The <code>with</code> statement uses it, most commonly for file handling.
<pre><code>with open("data.txt") as f:
    content = f.read()
# file is automatically closed here, even if read() raised an exception</code></pre></p>

<p><strong>Q28. How do you write a custom context manager?</strong><br/>
Implement a class with <code>__enter__</code>/<code>__exit__</code>, or use the <code>@contextmanager</code> decorator from <code>contextlib</code> with a generator function.
<pre><code>from contextlib import contextmanager

@contextmanager
def timer():
    import time; start = time.time()
    yield
    print(f"Elapsed: {time.time() - start:.2f}s")

with timer():
    sum(range(1000000))</code></pre></p>

<p><strong>Q29. What's the difference between append() and extend() for lists?</strong><br/>
<code>append(x)</code> adds x as a single element (even if x is a list, it's nested). <code>extend(iterable)</code> adds each element of the iterable individually.
<pre><code>a = [1, 2]; a.append([3, 4])   # [1, 2, [3, 4]]
b = [1, 2]; b.extend([3, 4])   # [1, 2, 3, 4]</code></pre></p>

<p><strong>Q30. What is the difference between a module and a package?</strong><br/>
A module is a single .py file. A package is a directory containing multiple modules plus an <code>__init__.py</code>, organizing related modules under one namespace.
<pre><code>import math          # module
import numpy as np   # package</code></pre></p>

<h3>OOP &amp; Advanced Topics (Q31–Q50)</h3>

<p><strong>Q31. Explain the four pillars of OOP with Python examples.</strong><br/>
Encapsulation (bundling data + methods, using <code>_</code>/<code>__</code> as a privacy convention), Abstraction (hiding implementation via abstract base classes), Inheritance (a class reusing another's behavior), Polymorphism (the same method name behaving differently per class).
<pre><code>class Animal:
    def speak(self): raise NotImplementedError

class Dog(Animal):
    def speak(self): return "Woof"

class Cat(Animal):
    def speak(self): return "Meow"

for a in [Dog(), Cat()]:
    print(a.speak())   # polymorphism: same call, different behavior</code></pre></p>

<p><strong>Q32. What's the difference between instance methods, class methods, and static methods?</strong><br/>
Instance methods (<code>self</code>) operate on a specific object. Class methods (<code>@classmethod</code>, <code>cls</code>) operate on the class itself, often used for alternate constructors. Static methods (<code>@staticmethod</code>) don't touch instance or class state — just a regular function namespaced inside the class.
<pre><code>class Pizza:
    def __init__(self, size): self.size = size

    @classmethod
    def large(cls): return cls(size=16)   # alternate constructor

    @staticmethod
    def is_valid_size(size): return size > 0</code></pre></p>

<p><strong>Q33. What are magic/dunder methods? Give examples.</strong><br/>
Special double-underscore methods that let objects integrate with Python's built-in syntax — <code>__init__</code> (constructor), <code>__str__</code> (print representation), <code>__len__</code> (len()), <code>__eq__</code> (==), <code>__add__</code> (+).
<pre><code>class Vector:
    def __init__(self, x, y): self.x, self.y = x, y
    def __add__(self, other): return Vector(self.x + other.x, self.y + other.y)
    def __repr__(self): return f"Vector({self.x}, {self.y})"

Vector(1, 2) + Vector(3, 4)   # Vector(4, 6)</code></pre></p>

<p><strong>Q34. What is Method Resolution Order (MRO) in multiple inheritance?</strong><br/>
MRO defines the order Python searches parent classes for a method/attribute in multiple inheritance, computed via the C3 linearization algorithm — inspectable via <code>ClassName.__mro__</code>.
<pre><code>class A:
    def greet(self): return "A"
class B(A):
    def greet(self): return "B"
class C(A):
    def greet(self): return "C"
class D(B, C): pass

D().greet()   # "B" — follows MRO: D, B, C, A</code></pre></p>

<p><strong>Q35. What is the difference between <code>__init__</code> and <code>__new__</code>?</strong><br/>
<code>__new__</code> actually creates and returns the new instance (called first). <code>__init__</code> initializes the already-created instance's attributes. Overriding <code>__new__</code> is rare — used for immutable types or singleton patterns.
<pre><code>class Singleton:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance</code></pre></p>

<p><strong>Q36. What is name mangling (<code>__attr</code>)?</strong><br/>
Prefixing an attribute with a double underscore (no trailing underscore) triggers name mangling — Python renames it internally to <code>_ClassName__attr</code>, making accidental override in subclasses harder (not truly private, just discouraged from casual access).
<pre><code>class Account:
    def __init__(self): self.__balance = 100

a = Account()
a.__balance            # AttributeError
a._Account__balance    # 100 — still accessible via the mangled name</code></pre></p>

<p><strong>Q37. What is the Global Interpreter Lock (GIL)?</strong><br/>
A mutex in CPython that lets only one thread execute Python bytecode at a time, even on multi-core machines — so CPU-bound multithreading doesn't get true parallelism, though I/O-bound multithreading still benefits since the GIL releases during I/O waits.<br/>
<em>Example:</em> For CPU-bound work, use <code>multiprocessing</code> (separate processes, each with its own GIL) instead of <code>threading</code> to get real parallel speedup.</p>

<p><strong>Q38. What's the difference between multithreading and multiprocessing?</strong><br/>
Threading shares memory but is limited by the GIL for CPU-bound tasks — good for I/O-bound work. Multiprocessing spawns separate processes with separate memory and their own GIL, achieving true parallelism for CPU-bound work at the cost of more memory and IPC overhead.
<pre><code>from multiprocessing import Pool
with Pool(4) as p:
    results = p.map(cpu_heavy_function, data_chunks)   # true parallel execution</code></pre></p>

<p><strong>Q39. What is duck typing?</strong><br/>
Python doesn't check an object's type before calling a method — "if it walks like a duck and quacks like a duck, it's a duck." Any object with the required method/attribute works, regardless of class hierarchy.
<pre><code>def make_it_speak(thing):
    return thing.speak()   # works for Dog, Cat, Robot — anything with a .speak() method</code></pre></p>

<p><strong>Q40. What is the difference between @property and a regular attribute?</strong><br/>
<code>@property</code> lets you define a method accessed like an attribute (no parentheses), enabling computed values or validation logic while keeping a clean attribute-style API.
<pre><code>class Circle:
    def __init__(self, radius): self.radius = radius
    @property
    def area(self): return 3.14159 * self.radius ** 2

c = Circle(5)
c.area   # 78.53975 — computed on access, called like an attribute</code></pre></p>

<p><strong>Q41. What are Python's scoping rules (LEGB)?</strong><br/>
Name lookup follows Local → Enclosing → Global → Built-in order — Python searches the innermost scope first, then outward, until it finds the name or raises NameError.
<pre><code>x = "global"
def outer():
    x = "enclosing"
    def inner():
        x = "local"
        print(x)   # "local"
    inner()</code></pre></p>

<p><strong>Q42. What do the <code>global</code> and <code>nonlocal</code> keywords do?</strong><br/>
<code>global</code> inside a function tells Python to modify a variable from the global scope instead of creating a local one. <code>nonlocal</code> does the same for the nearest enclosing (non-global) function scope — commonly used with closures.
<pre><code>counter = 0
def increment():
    global counter
    counter += 1   # modifies the global variable directly</code></pre></p>

<p><strong>Q43. What is the difference between sort() and sorted()?</strong><br/>
<code>list.sort()</code> sorts the list in place and returns None. <code>sorted(iterable)</code> returns a new sorted list, leaving the original unchanged, and works on any iterable — not just lists.
<pre><code>a = [3, 1, 2]; a.sort()          # a is now [1, 2, 3], returns None
b = [3, 1, 2]; c = sorted(b)      # b unchanged [3, 1, 2], c = [1, 2, 3]</code></pre></p>

<p><strong>Q44. What are Python virtual environments, and why use them?</strong><br/>
A virtual environment (<code>venv</code>, conda) creates an isolated Python installation with its own package set, so different projects can use different (possibly conflicting) library versions without interfering with each other or the system Python.
<pre><code>python -m venv myenv
source myenv/bin/activate
pip install pandas==1.5.3   # installed only inside this venv</code></pre></p>

<p><strong>Q45. What is the difference between <code>is None</code> and <code>== None</code>?</strong><br/>
<code>is None</code> checks identity against the single None singleton — the recommended, PEP 8 way. <code>== None</code> checks equality, which could theoretically be overridden by a custom <code>__eq__</code> and give unexpected results.
<pre><code>if x is None:   # preferred
    ...</code></pre></p>

<p><strong>Q46. How does Python handle default mutable arguments, and why is it a common bug?</strong><br/>
Default argument values are evaluated only once, at function definition time — so a mutable default (like a list) is shared across all calls that don't pass their own argument, causing unexpected accumulation.
<pre><code>def append_item(item, lst=[]):   # BUG: same list reused every call
    lst.append(item)
    return lst

append_item(1)   # [1]
append_item(2)   # [1, 2] — unexpected! Should default to None and create the list inside</code></pre></p>

<p><strong>Q47. What is <code>__slots__</code> used for?</strong><br/>
Defining <code>__slots__</code> on a class restricts instances to a fixed set of attributes (no <code>__dict__</code>), reducing per-instance memory overhead — useful when creating millions of small objects.
<pre><code>class Point:
    __slots__ = ('x', 'y')
    def __init__(self, x, y): self.x, self.y = x, y

p = Point(1, 2)
p.z = 5   # AttributeError — z not in __slots__</code></pre></p>

<p><strong>Q48. What's the difference between pickle and json for serialization?</strong><br/>
<code>json</code> produces human-readable, language-agnostic text but only supports basic types. <code>pickle</code> serializes almost any Python object (including custom classes) into binary, but it's Python-specific and unsafe to unpickle untrusted data (can execute arbitrary code).
<pre><code>import json, pickle
json.dumps({"a": 1})    # '{"a": 1}' — readable text
pickle.dumps({"a": 1})  # binary bytes, Python-only</code></pre></p>

<p><strong>Q49. What is list slicing, and how does step work?</strong><br/>
<code>list[start:stop:step]</code> extracts a sub-sequence; omitting start/stop defaults to the beginning/end, and a negative step reverses direction.
<pre><code>a = [0, 1, 2, 3, 4, 5]
a[1:4]    # [1, 2, 3]
a[::2]    # [0, 2, 4] — every 2nd element
a[::-1]   # [5, 4, 3, 2, 1, 0] — reversed</code></pre></p>

<p><strong>Q50. How would you read a large file without loading it all into memory?</strong><br/>
Iterate over the file object line by line (it's already lazy) instead of calling <code>.read()</code> or <code>.readlines()</code>, which load the entire file at once.
<pre><code>with open("huge_log.txt") as f:
    for line in f:        # reads one line at a time
        process(line)</code></pre></p>
`
  },
];
