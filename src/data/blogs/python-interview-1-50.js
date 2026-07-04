export const python_interview_1_50 = {
  slug: 'python-interview-1-50',
  title: 'Python ইন্টারভিউ প্রশ্ন (Beginner → Intermediate) — ৫০টি প্রশ্ন উদাহরণসহ',
  description: 'Python-এর core language basics, data structure, OOP, decorator, generator এবং memory management নিয়ে ৫০টি গুরুত্বপূর্ণ ইন্টারভিউ প্রশ্ন — প্রতিটির সাথে code example।',
  date: 'জুলাই ২০২৬',
  category: 'Python ইন্টারভিউ প্রশ্ন',
  readTime: 30,
  content: `
    <h3>Core Python Basics (Q1–Q15)</h3>

    <p><strong>Q1. List এবং tuple-এর মধ্যে পার্থক্য কী?</strong><br/>
    List mutable (element add/remove/change করা যায়)। Tuple immutable, একটু দ্রুত, এবং dict key বা set member হিসেবে ব্যবহার করা যায় (সব element hashable হলে) — list এটা পারে না।
    <pre><code>my_list = [1, 2, 3]
my_list.append(4)          # ঠিকভাবে কাজ করে

my_tuple = (1, 2, 3)
my_tuple[0] = 99            # TypeError: tuple item assignment support করে না</code></pre></p>

    <p><strong>Q2. == এবং is-এর মধ্যে পার্থক্য কী?</strong><br/>
    == value compare করে (__eq__ call করে)। is object identity compare করে (একই memory address কিনা)। একই value থাকা দুটো object আলাদাও হতে পারে।
    <pre><code>a = [1, 2, 3]
b = [1, 2, 3]
print(a == b)   # True  (value একই)
print(a is b)   # False (আলাদা object)
print(a is a)   # True  (একই object)</code></pre></p>

    <p><strong>Q3. Python-এ mutable এবং immutable type কী কী?</strong><br/>
    Mutable: list, dict, set (in place বদলানো যায়)। Immutable: int, float, str, tuple, frozenset, bool — এদের কোনো "change" আসলে নতুন object তৈরি করে।
    <pre><code>s = "hello"
s[0] = 'H'          # TypeError — string immutable
s = 'H' + s[1:]     # এটা একটা সম্পূর্ণ নতুন string object তৈরি করে</code></pre></p>

    <p><strong>Q4. Python-এর dynamic typing explain করুন।</strong><br/>
    Variable কোনো fixed type-এ bound থাকে না — একই নাম runtime-এ যেকোনো type-এ reassign করা যায়, এবং type check হয় runtime-এ, compile time-এ নয়।
    <pre><code>x = 5
x = "now a string"
x = [1, 2, 3]       # কোনো error নেই — x-এর type প্রতিবার বদলে যায়</code></pre></p>

    <p><strong>Q5. Deep copy এবং shallow copy-এর মধ্যে পার্থক্য কী?</strong><br/>
    Shallow copy বাইরের container copy করে, কিন্তু nested object-গুলো এখনো shared reference থাকে। Deep copy সবকিছু recursively copy করে, তাই nested object সম্পূর্ণ independent হয়ে যায়।
    <pre><code>import copy
a = [[1, 2], [3, 4]]
b = copy.copy(a)
b[0][0] = 99         # a[0][0]-ও 99 হয়ে যায় — inner list shared!

c = copy.deepcopy(a)
c[0][0] = 100        # a সম্পূর্ণ unaffected থাকে</code></pre></p>

    <p><strong>Q6. *args এবং **kwargs কী?</strong><br/>
    *args extra positional argument-গুলোকে একটা tuple-এ জমা করে। **kwargs extra keyword argument-গুলোকে একটা dict-এ জমা করে — variable সংখ্যক argument নেওয়া function-এ ব্যবহার হয়।
    <pre><code>def f(*args, **kwargs):
    print(args, kwargs)

f(1, 2, a=3, b=4)   # (1, 2) {'a': 3, 'b': 4}</code></pre></p>

    <p><strong>Q7. Python কি "pass by value" নাকি "pass by reference"?</strong><br/>
    ঠিক কোনোটাই না — Python object reference-কে value দিয়ে pass করে ("pass by object reference")। Immutable argument-এর ক্ষেত্রে function caller-এর variable বদলাতে পারে না। Mutable argument-এর ক্ষেত্রে function object-টা in place modify করতে পারে, কিন্তু parameter name reassign করলে caller-এর variable-এ প্রভাব পড়ে না।
    <pre><code>def modify(lst): lst.append(4)
def reassign(lst): lst = [100]

a = [1, 2, 3]
modify(a)     # a হয়ে যায় [1, 2, 3, 4] — in-place mutation দেখা যায়
reassign(a)   # a থেকে যায় [1, 2, 3, 4] — শুধু local name rebind হয়েছে</code></pre></p>

    <p><strong>Q8. Lambda function কী?</strong><br/>
    একটা anonymous, single-expression function যা inline define করা হয় — সাধারণত sorting, filtering, বা mapping-এর জন্য একটা ছোট callback হিসেবে ব্যবহৃত হয়।
    <pre><code>pairs = [(1, 'b'), (2, 'a')]
sorted(pairs, key=lambda x: x[1])   # দ্বিতীয় element দিয়ে sort করে</code></pre></p>

    <p><strong>Q9. List comprehension বনাম generator expression explain করুন।</strong><br/>
    List comprehension [x for x in ...] পুরো list সাথে সাথে memory-তে তৈরি করে। Generator expression (x for x in ...) lazily, একটা করে item তৈরি করে — বড় sequence-এর জন্য অনেক কম memory ব্যবহার করে।
    <pre><code>squares_list = [x**2 for x in range(1000000)]   # পুরো list memory-তে তৈরি করে
squares_gen  = (x**2 for x in range(1000000))   # lazy — consume না করা পর্যন্ত প্রায় zero memory</code></pre></p>

    <p><strong>Q10. List-এ remove(), pop(), এবং del-এর মধ্যে পার্থক্য কী?</strong><br/>
    remove(value) প্রথম matching value সরিয়ে দেয়। pop(index) সেই index-এর item সরিয়ে return করে (default: last)। del list[index] value return না করেই index অনুযায়ী সরায়; del পুরো variable-ও delete করতে পারে।
    <pre><code>lst = [1, 2, 3]
lst.remove(2)   # [1, 3]
lst.pop()       # 3 return করে, lst হয়ে যায় [1]
del lst[0]      # index 0 সরিয়ে দেয়</code></pre></p>

    <p><strong>Q11. Python-এর built-in data type কী কী?</strong><br/>
    Numeric (int, float, complex), sequence (str, list, tuple, range), mapping (dict), set type (set, frozenset), boolean (bool), এবং None type।
    <pre><code>type(5), type(5.0), type("a"), type([1]), type({})
# int, float, str, list, dict</code></pre></p>

    <p><strong>Q12. Python-এ string interning কী?</strong><br/>
    CPython ছোট string ও identifier automatically cache (intern) করে, তাই identical string literal একটা optimization হিসেবে memory-তে একই object-কে point করতে পারে — এটা string-এ is comparison-কে প্রভাবিত করে কিন্তু correctness-এর জন্য এর উপর নির্ভর করা উচিত না।
    <pre><code>a = "hello"
b = "hello"
print(a is b)   # interning-এর কারণে প্রায়ই True — কিন্তু সব string-এ guaranteed নয়</code></pre></p>

    <p><strong>Q13. <code>pass</code> statement কী করে?</strong><br/>
    একটা no-op placeholder statement, যেখানে Python-এর syntax একটা statement দাবি করে কিন্তু এখনো কোনো action দরকার নেই — যেমন empty function body বা stub class।
    <pre><code>def todo_later():
    pass   # পরে implement করব</code></pre></p>

    <p><strong>Q14. <code>break</code>, <code>continue</code>, এবং <code>pass</code>-এর মধ্যে পার্থক্য কী?</strong><br/>
    break পুরো loop থেকে বেরিয়ে আসে। continue পরের iteration-এ skip করে। pass কিছুই করে না — শুধু syntax satisfy করে।
    <pre><code>for i in range(5):
    if i == 2: continue   # 2 skip করে, loop চালিয়ে যায়
    if i == 4: break       # 4-এ থেমে যায়
    print(i)                # print করে 0, 1, 3</code></pre></p>

    <p><strong>Q15. Python কীভাবে memory manage করে?</strong><br/>
    Reference counting-এর মাধ্যমে (কোনো object-এর reference count zero হওয়ার সাথে সাথেই free হয়ে যায়) এবং একটা generational garbage collector-এর মাধ্যমে যা reference counting একা ধরতে পারে না এমন reference cycle detect ও clean করে।
    <pre><code>import gc
gc.collect()   # unreachable reference cycle clean করতে collection cycle force করে</code></pre></p>

    <h3>Data Structures &amp; Functions (Q16–Q30)</h3>

    <p><strong>Q16. Set এবং list-এর মধ্যে পার্থক্য কী?</strong><br/>
    Set unique, unordered element রাখে, O(1) average membership testing সহ। List duplicate allow করে, insertion order বজায় রাখে, এবং membership testing O(n)।
    <pre><code>list(set([1, 2, 2, 3, 3, 3]))   # [1, 2, 3] — duplicate সরানো হয়েছে</code></pre></p>

    <p><strong>Q17. Dictionary comprehension কী?</strong><br/>
    একটা concise {k: v for ...} syntax যা একটা iterable থেকে dict তৈরি করে — list comprehension-এর মতোই।
    <pre><code>{x: x**2 for x in range(5)}
# {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}</code></pre></p>

    <p><strong>Q18. Iterable-এর জন্য Python-এর * unpacking operator explain করুন।</strong><br/>
    * একটা iterable-এর element-গুলোকে individual item-এ unpack করে — function call, assignment, এবং collection merge করার সময় ব্যবহার হয়।
    <pre><code>a, *rest = [1, 2, 3, 4]     # a = 1, rest = [2, 3, 4]
merged = [*[1, 2], *[3, 4]]  # [1, 2, 3, 4]</code></pre></p>

    <p><strong>Q19. Python-এ decorator কী?</strong><br/>
    একটা decorator হলো এমন একটা function যা অন্য একটা function-কে wrap করে তার source code বদলানো ছাড়াই behavior extend করে — @decorator syntax দিয়ে implement করা হয়।
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

    <p><strong>Q20. Closure কী?</strong><br/>
    Closure হলো একটা nested function যা তার enclosing function-এর scope-এর variable মনে রাখে এবং access করতে পারে, এমনকি outer function return হয়ে যাওয়ার পরেও।
    <pre><code>def make_multiplier(n):
    def multiply(x):
        return x * n
    return multiply

times3 = make_multiplier(3)
times3(5)   # 15 — n=3 মনে রেখেছে</code></pre></p>

    <p><strong>Q21. Function এবং generator function-এর মধ্যে পার্থক্য কী?</strong><br/>
    একটা সাধারণ function return দিয়ে একটা মাত্র value দেয় এবং সেখানেই execution শেষ হয়। Generator function yield ব্যবহার করে, execution pause ও resume করে, lazily একটা করে value-এর sequence তৈরি করে।
    <pre><code>def gen():
    yield 1
    yield 2
    yield 3

for val in gen():
    print(val)   # 1, 2, 3 — একটা করে, সব একসাথে memory-তে না রেখে</code></pre></p>

    <p><strong>Q22. Iterable এবং iterator-এর মধ্যে পার্থক্য কী?</strong><br/>
    Iterable এমন যেকোনো object যার উপর loop চালানো যায় (__iter__ implement করে), যেমন list। Iterator হলো iter()-কে একটা iterable-এ call করলে যা return হয় — এটা __next__ implement করে এবং position track করে, exhausted হলে StopIteration raise করে।
    <pre><code>lst = [1, 2, 3]   # iterable
it = iter(lst)     # iterator
next(it)   # 1
next(it)   # 2</code></pre></p>

    <p><strong>Q23. map(), filter(), এবং reduce() explain করুন।</strong><br/>
    map(func, iterable) প্রতিটা element-এ func apply করে। filter(func, iterable) শুধু সেই element রাখে যেখানে func True return করে। reduce(func, iterable) (functools থেকে) element-গুলোকে cumulatively একটা single value-এ combine করে।
    <pre><code>from functools import reduce
nums = [1, 2, 3, 4]
list(map(lambda x: x * 2, nums))        # [2, 4, 6, 8]
list(filter(lambda x: x % 2 == 0, nums)) # [2, 4]
reduce(lambda a, b: a + b, nums)         # 10</code></pre></p>

    <p><strong>Q24. Exception handling কী? try/except/else/finally explain করুন।</strong><br/>
    try-তে এমন code থাকে যা exception raise করতে পারে। except নির্দিষ্ট exception type catch করে। else শুধু তখনই run হয় যখন কোনো exception হয়নি। finally সবসময় run হয় — cleanup-এর জন্য (file close করা, resource release করা)।
    <pre><code>try:
    result = 10 / 0
except ZeroDivisionError as e:
    print("Cannot divide by zero:", e)
else:
    print("No error occurred")
finally:
    print("This always runs")</code></pre></p>

    <p><strong>Q25. Exception এবং BaseException-এর মধ্যে পার্থক্য কী?</strong><br/>
    BaseException হলো exception hierarchy-র root, যাতে SystemExit, KeyboardInterrupt, এবং GeneratorExit-ও আছে। Exception হলো একটা subclass যা সাধারণ error cover করে যা আপনার code সাধারণত catch করা উচিত — bare Exception catch করা (BaseException নয়) best practice, যাতে ভুলবশত Ctrl+C বা system exit swallow না হয়ে যায়।
    <pre><code>try:
    risky_operation()
except Exception as e:   # ValueError, TypeError ইত্যাদি catch করে — KeyboardInterrupt নয়
    print(e)</code></pre></p>

    <p><strong>Q26. Custom exception কীভাবে তৈরি করবেন?</strong><br/>
    Built-in Exception class-কে (বা আরও specific কোনো একটা) subclass করুন, চাইলে __init__ override করে custom attribute যোগ করুন।
    <pre><code>class InsufficientFundsError(Exception):
    def __init__(self, balance, amount):
        super().__init__(f"Cannot withdraw {amount}, balance is {balance}")
        self.balance = balance

raise InsufficientFundsError(100, 500)</code></pre></p>

    <p><strong>Q27. Context manager এবং <code>with</code> statement কী?</strong><br/>
    একটা context manager guarantee দেয় যে setup/teardown code একটা block-এর চারপাশে run হবে, exception হলেও — __enter__/__exit__ দিয়ে implement করা হয়। with statement এটা ব্যবহার করে, সবচেয়ে বেশি file handling-এ।
    <pre><code>with open("data.txt") as f:
    content = f.read()
# read() exception raise করলেও file এখানে automatically close হয়ে যায়</code></pre></p>

    <p><strong>Q28. কীভাবে একটা custom context manager লিখবেন?</strong><br/>
    __enter__/__exit__ সহ একটা class implement করুন, বা contextlib থেকে @contextmanager decorator একটা generator function-এর সাথে ব্যবহার করুন।
    <pre><code>from contextlib import contextmanager

@contextmanager
def timer():
    import time; start = time.time()
    yield
    print(f"Elapsed: {time.time() - start:.2f}s")

with timer():
    sum(range(1000000))</code></pre></p>

    <p><strong>Q29. List-এ append() এবং extend()-এর মধ্যে পার্থক্য কী?</strong><br/>
    append(x) x-কে একটা single element হিসেবে যোগ করে (x list হলেও nested হয়ে যায়)। extend(iterable) iterable-এর প্রতিটা element আলাদা করে যোগ করে।
    <pre><code>a = [1, 2]; a.append([3, 4])   # [1, 2, [3, 4]]
b = [1, 2]; b.extend([3, 4])   # [1, 2, 3, 4]</code></pre></p>

    <p><strong>Q30. Module এবং package-এর মধ্যে পার্থক্য কী?</strong><br/>
    Module হলো একটা single .py file। Package হলো একটা directory যাতে অনেকগুলো module এবং একটা __init__.py থাকে, related module-গুলোকে এক namespace-এর নিচে organize করে।
    <pre><code>import math          # module
import numpy as np   # package</code></pre></p>

    <h3>OOP &amp; Advanced Topics (Q31–Q50)</h3>

    <p><strong>Q31. Python উদাহরণসহ OOP-এর চারটা pillar explain করুন।</strong><br/>
    Encapsulation (data + method একসাথে bundle করা, privacy convention হিসেবে _ বা __ ব্যবহার), Abstraction (abstract base class দিয়ে implementation লুকানো), Inheritance (একটা class অন্যটার behavior reuse করে), Polymorphism (একই method name প্রতিটা class-এ ভিন্নভাবে behave করে)।
    <pre><code>class Animal:
    def speak(self): raise NotImplementedError

class Dog(Animal):
    def speak(self): return "Woof"

class Cat(Animal):
    def speak(self): return "Meow"

for a in [Dog(), Cat()]:
    print(a.speak())   # polymorphism: একই call, ভিন্ন behavior</code></pre></p>

    <p><strong>Q32. Instance method, class method, এবং static method-এর মধ্যে পার্থক্য কী?</strong><br/>
    Instance method (self) একটা নির্দিষ্ট object-এর উপর কাজ করে। Class method (@classmethod, cls) class-এর উপর কাজ করে, প্রায়ই alternate constructor-এর জন্য ব্যবহার হয়। Static method (@staticmethod) instance বা class state touch করে না — class-এর ভেতরে namespace করা একটা সাধারণ function মাত্র।
    <pre><code>class Pizza:
    def __init__(self, size): self.size = size

    @classmethod
    def large(cls): return cls(size=16)   # alternate constructor

    @staticmethod
    def is_valid_size(size): return size > 0</code></pre></p>

    <p><strong>Q33. Magic/dunder method কী? উদাহরণ দিন।</strong><br/>
    Double-underscore দিয়ে ঘেরা special method যা object-কে Python-এর built-in syntax-এর সাথে integrate করতে দেয় — __init__ (constructor), __str__ (print representation), __len__ (len()), __eq__ (==), __add__ (+)।
    <pre><code>class Vector:
    def __init__(self, x, y): self.x, self.y = x, y
    def __add__(self, other): return Vector(self.x + other.x, self.y + other.y)
    def __repr__(self): return f"Vector({self.x}, {self.y})"

Vector(1, 2) + Vector(3, 4)   # Vector(4, 6)</code></pre></p>

    <p><strong>Q34. Multiple inheritance-এ Method Resolution Order (MRO) কী?</strong><br/>
    MRO নির্ধারণ করে multiple inheritance-এ Python কোন order-এ parent class-এ method/attribute খুঁজবে, C3 linearization algorithm দিয়ে calculate করা হয় — ClassName.__mro__ দিয়ে দেখা যায়।
    <pre><code>class A:
    def greet(self): return "A"
class B(A):
    def greet(self): return "B"
class C(A):
    def greet(self): return "C"
class D(B, C): pass

D().greet()   # "B" — MRO অনুসরণ করে: D, B, C, A</code></pre></p>

    <p><strong>Q35. __init__ এবং __new__-এর মধ্যে পার্থক্য কী?</strong><br/>
    __new__ আসলে নতুন instance তৈরি করে return করে (প্রথমে call হয়)। __init__ ইতিমধ্যে তৈরি হওয়া instance-এর attribute initialize করে। __new__ override করা rare — immutable type বা singleton pattern-এ ব্যবহৃত।
    <pre><code>class Singleton:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance</code></pre></p>

    <p><strong>Q36. Name mangling (__attr) কী?</strong><br/>
    একটা attribute-এর আগে double underscore (শেষে underscore ছাড়া) দিলে name mangling হয় — Python internally এটাকে _ClassName__attr-এ rename করে, subclass-এ accidental override করা কঠিন করে (সত্যিকারের private নয়, শুধু casual access থেকে discourage করে)।
    <pre><code>class Account:
    def __init__(self): self.__balance = 100

a = Account()
a.__balance            # AttributeError
a._Account__balance    # 100 — mangled name দিয়ে এখনো access করা যায়</code></pre></p>

    <p><strong>Q37. Global Interpreter Lock (GIL) কী?</strong><br/>
    CPython-এ একটা mutex যা একবারে শুধু একটা thread-কে Python bytecode execute করতে দেয়, multi-core machine-এও — এর মানে CPU-bound multithreading-এ Python-এ সত্যিকারের parallelism পাওয়া যায় না, যদিও I/O-bound multithreading এখনো লাভবান হয় কারণ I/O wait-এর সময় GIL release হয়।<br/>
    উদাহরণ: CPU-bound কাজের জন্য threading-এর বদলে multiprocessing ব্যবহার করুন (আলাদা process, প্রতিটার নিজের GIL) সত্যিকারের parallel speedup পেতে।</p>

    <p><strong>Q38. Multithreading এবং multiprocessing-এর মধ্যে পার্থক্য কী?</strong><br/>
    Threading memory share করে কিন্তু CPU-bound task-এ GIL দ্বারা সীমাবদ্ধ — I/O-bound কাজের জন্য ভালো। Multiprocessing আলাদা memory ও নিজের GIL সহ আলাদা process spawn করে, CPU-bound কাজে সত্যিকারের parallelism achieve করে, বেশি memory ও IPC overhead-এর বিনিময়ে।
    <pre><code>from multiprocessing import Pool
with Pool(4) as p:
    results = p.map(cpu_heavy_function, data_chunks)   # সত্যিকারের parallel execution</code></pre></p>

    <p><strong>Q39. Duck typing কী?</strong><br/>
    Python কোনো method call করার আগে object-এর type check করে না — "যদি হাঁসের মতো হাঁটে আর হাঁসের মতো ডাকে, তবে এটা হাঁস।" দরকারি method/attribute থাকা যেকোনো object কাজ করে, class hierarchy যাই হোক না কেন।
    <pre><code>def make_it_speak(thing):
    return thing.speak()   # Dog, Cat, Robot — .speak() method থাকা যেকোনো কিছুর জন্য কাজ করে</code></pre></p>

    <p><strong>Q40. @property এবং একটা সাধারণ attribute-এর মধ্যে পার্থক্য কী?</strong><br/>
    @property একটা method define করতে দেয় যা attribute-এর মতো access হয় (parenthesis ছাড়া), computed value বা validation logic সম্ভব করে clean attribute-style API বজায় রেখে।
    <pre><code>class Circle:
    def __init__(self, radius): self.radius = radius
    @property
    def area(self): return 3.14159 * self.radius ** 2

c = Circle(5)
c.area   # 78.53975 — access করার সময় calculate হয়, attribute-এর মতো call হয়</code></pre></p>

    <p><strong>Q41. Python-এর scoping rule (LEGB) কী?</strong><br/>
    Name lookup Local → Enclosing → Global → Built-in order অনুসরণ করে — Python প্রথমে সবচেয়ে ভেতরের scope খোঁজে, তারপর বাইরের দিকে, যতক্ষণ না name খুঁজে পায় বা NameError raise করে।
    <pre><code>x = "global"
def outer():
    x = "enclosing"
    def inner():
        x = "local"
        print(x)   # "local"
    inner()</code></pre></p>

    <p><strong>Q42. <code>global</code> এবং <code>nonlocal</code> keyword কী করে?</strong><br/>
    function-এর ভেতরে global বললে Python-কে local variable তৈরি না করে global scope-এর variable modify করতে বলে। nonlocal একই কাজ করে সবচেয়ে কাছের enclosing (non-global) function scope-এর জন্য — closure-এর সাথে সাধারণত ব্যবহৃত হয়।
    <pre><code>counter = 0
def increment():
    global counter
    counter += 1   # সরাসরি global variable modify করে</code></pre></p>

    <p><strong>Q43. sort() এবং sorted()-এর মধ্যে পার্থক্য কী?</strong><br/>
    list.sort() list-কে in place sort করে এবং None return করে। sorted(iterable) একটা নতুন sorted list return করে, original অপরিবর্তিত রেখে, এবং list ছাড়াও যেকোনো iterable-এর জন্য কাজ করে।
    <pre><code>a = [3, 1, 2]; a.sort()          # a এখন [1, 2, 3], None return করে
b = [3, 1, 2]; c = sorted(b)      # b অপরিবর্তিত [3, 1, 2], c = [1, 2, 3]</code></pre></p>

    <p><strong>Q44. Python virtual environment কী এবং কেন ব্যবহার করবেন?</strong><br/>
    একটা virtual environment (venv, conda) নিজের package set-সহ একটা isolated Python installation তৈরি করে, তাই বিভিন্ন project ভিন্ন (সম্ভবত conflicting) library version ব্যবহার করতে পারে একে অপরকে বা system Python-কে প্রভাবিত না করেই।
    <pre><code>python -m venv myenv
source myenv/bin/activate
pip install pandas==1.5.3   # শুধু এই venv-এর ভেতরে install হয়</code></pre></p>

    <p><strong>Q45. <code>is None</code> এবং <code>== None</code>-এর মধ্যে পার্থক্য কী?</strong><br/>
    is None একমাত্র None singleton object-এর সাথে identity check করে — recommended, PEP 8 way। == None equality check করে, যা theoretically একটা custom __eq__ দিয়ে override হয়ে অপ্রত্যাশিত result দিতে পারে।
    <pre><code>if x is None:   # preferred
    ...</code></pre></p>

    <p><strong>Q46. Python default mutable argument কীভাবে handle করে, এবং এটা কেন একটা common bug?</strong><br/>
    Default argument value শুধু একবারই evaluate হয়, function definition time-এ — তাই একটা mutable default (যেমন list) নিজের argument pass না করা সব call-এর মধ্যে shared থাকে, call জুড়ে অপ্রত্যাশিত accumulation তৈরি করে।
    <pre><code>def append_item(item, lst=[]):   # BUG: প্রতিবার একই list reuse হয়
    lst.append(item)
    return lst

append_item(1)   # [1]
append_item(2)   # [1, 2] — অপ্রত্যাশিত! default None রেখে ভেতরে list তৈরি করা উচিত</code></pre></p>

    <p><strong>Q47. __slots__ কীসের জন্য ব্যবহার হয়?</strong><br/>
    একটা class-এ __slots__ define করলে instance-গুলো একটা fixed attribute set-এ সীমাবদ্ধ থাকে (__dict__ নেই), per-instance memory overhead কমে — লক্ষ লক্ষ ছোট object তৈরি করার সময় দরকারি।
    <pre><code>class Point:
    __slots__ = ('x', 'y')
    def __init__(self, x, y): self.x, self.y = x, y

p = Point(1, 2)
p.z = 5   # AttributeError — z, __slots__-এ নেই</code></pre></p>

    <p><strong>Q48. Serialization-এর জন্য pickle এবং json-এর মধ্যে পার্থক্য কী?</strong><br/>
    json human-readable, language-agnostic text তৈরি করে কিন্তু শুধু basic type support করে। pickle প্রায় যেকোনো Python object (custom class-সহ) binary format-এ serialize করে, কিন্তু এটা Python-specific এবং untrusted data unpickle করা unsafe (arbitrary code execute করতে পারে)।
    <pre><code>import json, pickle
json.dumps({"a": 1})    # '{"a": 1}' — readable text
pickle.dumps({"a": 1})  # binary bytes, শুধু Python</code></pre></p>

    <p><strong>Q49. List slicing কী এবং step কীভাবে কাজ করে?</strong><br/>
    list[start:stop:step] একটা sub-sequence বের করে; start/stop বাদ দিলে default শুরু/শেষ হয়, এবং negative step direction reverse করে দেয়।
    <pre><code>a = [0, 1, 2, 3, 4, 5]
a[1:4]    # [1, 2, 3]
a[::2]    # [0, 2, 4] — প্রতি ২য় element
a[::-1]   # [5, 4, 3, 2, 1, 0] — reversed</code></pre></p>

    <p><strong>Q50. পুরো file memory-তে load না করে বড় file কীভাবে পড়বেন?</strong><br/>
    .read() বা .readlines() call করার বদলে (যা একবারে পুরো file load করে) file object-টার উপর line by line iterate করুন (এটা এমনিতেই lazy)।
    <pre><code>with open("huge_log.txt") as f:
    for line in f:        # একবারে একটা line পড়ে
        process(line)</code></pre></p>
  `
};
