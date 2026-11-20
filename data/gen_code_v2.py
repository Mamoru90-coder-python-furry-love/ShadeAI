import sys, random
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
qa = []
def add(q, a): qa.append('USER: ' + q); qa.append('AI: ' + a)

# Template code patterns (without backticks)
langs = ['Python', 'JavaScript', 'C++', 'Java', 'C#', 'Go', 'Rust', 'TypeScript', 'Kotlin', 'Swift', 'Scala', 'Haskell', 'Lua', 'Ruby', 'PHP', 'R', 'Julia', 'Dart', 'SQL', 'Elixir', 'Prolog']

def gen_py(code): return 'Python: ' + code
def gen_js(code): return 'JavaScript: ' + code
def gen_cpp(code): return 'C++: ' + code

# Generate patterns
for lang in langs:
    # Hello World
    hello_map = {
        'Python': 'print("Hello World")',
        'JavaScript': 'console.log("Hello World")',
        'C++': '#include <iostream>\nint main() { std::cout << "Hello World"; return 0; }',
        'Java': 'public class Hello {\n  public static void main(String[] args) {\n    System.out.println("Hello World");\n  }\n}',
        'C#': 'using System;\nclass Program { static void Main() { Console.WriteLine("Hello World"); } }',
        'Go': 'package main\nimport "fmt"\nfunc main() { fmt.Println("Hello World") }',
        'Rust': 'fn main() { println!("Hello World"); }',
        'Kotlin': 'fun main() = println("Hello World")',
        'Swift': 'print("Hello World")',
        'Scala': 'object Hello extends App { println("Hello World") }',
        'Haskell': 'main = putStrLn "Hello World"',
        'Lua': 'print("Hello World")',
        'Ruby': 'puts "Hello World"',
        'PHP': '<?php echo "Hello World"; ?>',
        'R': 'cat("Hello World")',
        'Julia': 'println("Hello World")',
        'Dart': 'void main() { print("Hello World"); }',
        'Elixir': 'IO.puts("Hello World")',
        'SQL': "SELECT 'Hello World' AS greeting;",
        'Prolog': "write('Hello World'), nl.",
        'TypeScript': 'console.log("Hello World")',
    }
    code = hello_map.get(lang, 'print("Hello World")')
    add(f'Напиши Hello World на {lang}', code)
    add(f'Як написати Hello World в {lang}?', code)
    
    # Calculator
    calc_map = {
        'Python': 'def calc(a, op, b):\n  ops = {"+": lambda x,y: x+y, "-": lambda x,y: x-y, "*": lambda x,y: x*y, "/": lambda x,y: x/y if y else "err"}\n  return ops.get(op, lambda x,y: "unknown")(a, b)',
        'JavaScript': 'function calc(a, op, b) {\n  const ops = {"+": (x,y)=>x+y, "-": (x,y)=>x-y, "*": (x,y)=>x*y, "/": (x,y)=>x/y}\n  return ops[op] ? ops[op](a,b) : "unknown"\n}',
        'C++': 'double calc(double a, char op, double b) {\n  switch(op) {\n    case "+": return a+b; case "-": return a-b;\n    case "*": return a*b; case "/": return b ? a/b : 0;\n    default: return 0;\n  }\n}',
        'Java': 'double calc(double a, String op, double b) {\n  switch(op) {\n    case "+": return a+b; case "-": return a-b;\n    case "*": return a*b; case "/": return b != 0 ? a/b : 0;\n    default: return 0;\n  }\n}',
        'Rust': 'fn calc(a: f64, op: &str, b: f64) -> f64 {\n  match op {\n    "+" => a+b, "-" => a-b, "*" => a*b,\n    "/" => if b != 0.0 { a/b } else { 0.0 }, _ => 0.0\n  }\n}',
        'Go': 'func calc(a, b float64, op string) float64 {\n  switch op {\n    case "+": return a+b; case "-": return a-b;\n    case "*": return a*b; case "/": if b != 0 { return a/b }\n  }\n  return 0\n}',
        'SQL': "SELECT CASE op WHEN '+' THEN a+b WHEN '-' THEN a-b WHEN '*' THEN a*b WHEN '/' THEN a/b END FROM calc;",
    }
    code = calc_map.get(lang, 'function calc(a, op, b) { if (op == "+") return a+b; if (op == "-") return a-b; if (op == "*") return a*b; if (op == "/") return a/b }')
    add(f'Напиши калькулятор на {lang}', code)
    
    # Factorial
    fact_map = {
        'Python': 'def fact(n): return 1 if n <= 1 else n * fact(n-1)',
        'JavaScript': 'function fact(n) { return n <= 1 ? 1 : n * fact(n-1) }',
        'C++': 'int fact(int n) { return n <= 1 ? 1 : n * fact(n-1); }',
        'Java': 'int fact(int n) { return n <= 1 ? 1 : n * fact(n-1); }',
        'Rust': 'fn fact(n: u64) -> u64 { if n <= 1 { 1 } else { n * fact(n-1) } }',
        'Haskell': 'fact 0 = 1\nfact n = n * fact (n-1)',
        'Elixir': 'def fact(0), do: 1\ndef fact(n), do: n * fact(n-1)',
        'Prolog': 'fact(0, 1).\nfact(N, R) :- N > 0, N1 is N-1, fact(N1, R1), R is N * R1.',
        'SQL': "WITH RECURSIVE fact(n,f) AS (SELECT 0,1 UNION ALL SELECT n+1, f*(n+1) FROM fact WHERE n<10) SELECT * FROM fact",
        'Scala': 'def fact(n: Int): Int = if (n <= 1) 1 else n * fact(n-1)',
    }
    code = fact_map.get(lang, 'def fact(n): return 1 if n <= 1 else n * fact(n-1)')
    add(f'Напиши факторіал на {lang}', code)
    add(f'Як зробити факторіал в {lang}?', code)

# General CS questions (diverse starts, no backticks)
cs = [
    ("Що таке O(n)?", "O(n) означає лінійну складність. Час виконання пропорційний розміру даних. Приклад: пошук в масиві."),
    ("Що таке рекурсія?", "Рекурсія — функція викликає себе. Потрібен базовий випадок, інакше буде stack overflow. Приклад: факторіал через рекурсію."),
    ("Що таке масив?", "Масив зберігає елементи одного типу за індексом. Доступ O(1). В Python це list, в JS Array."),
    ("Що таке хеш-таблиця?", "Структура ключ-значення. Доступ O(1). В Python dict, в Java HashMap, в JS Map і Object."),
    ("Що таке стек?", "Стек LIFO. Операції push(додати) і pop(видалити) з вершини за O(1). Приклад: стек викликів функцій."),
    ("Що таке черга?", "Черга FIFO. dodavaty в кінець, забирати з початку. Використовується в BFS."),
    ("Що таке ООП?", "Об'єктно-орієнтоване програмування. Чотири принципи: інкапсуляція, наслідування, поліморфізм, абстракція."),
    ("Що таке SQL?", "Мова запитів до баз даних. SELECT FROM WHERE, JOIN, GROUP BY, ORDER BY. Розрізняють DDL, DML, DCL."),
    ("Що таке git?", "Контроль версій. git init, add, commit, push, pull, branch, merge. GitHub для віддаленого зберігання."),
    ("Що таке JSON?", "Формат обміну даними: { 'key': 'value', 'num': 42, 'arr': [1,2,3] }. Легко читається, використовується в API."),
    ("Що таке HTTP статуси?", "1xx інформація, 2xx успіх (200 OK), 3xx редирект, 4xx помилка клієнта (404 Not Found), 5xx помилка сервера (500)."),
    ("Що таке Docker?", "Контейнеризація. Dockerfile описує середовище, docker build збирає образ, docker run запускає контейнер."),
    ("Що таке REST?", "Архітектура API. Ресурси з CRUD операціями: GET, POST, PUT, DELETE. Статуси 200, 201, 204, 400, 404."),
    ("Що таке TCP/IP?", "TCP надійна передача з'єднання (HTTP, FTP). IP маршрутизація пакетів між мережами."),
    ("Що таке лямбда?", "Анонімна функція. Python: lambda x: x*2. JS: x => x*2. Java: (x) -> x*2. C#: x => x*2."),
    ("Що таке регекс?", "Регулярні вирази для пошуку. \\d цифра, \\w буква/цифра, * 0+, + 1+. Приклад: [A-Z][a-z]+ імена."),
    ("Що таке криптографія?", "Шифрування. Симетричне AES (один ключ), асиметричне RSA (публічний/приватний ключ), хешування SHA."),
    ("Що таке dependency injection?", "Залежності передаються ззовні, не створюються всередині. Легше тестувати і змінювати."),
    ("Що таке linked list?", "Вузли з даними і посиланням на наступний. Доступ O(n), але вставка/видалення O(1) в середині."),
    ("Що таке бінарне дерево?", "Кожен вузол має максимум 2 дітей. BST: ліве < корінь < праве. Пошук O(log n) в середньому."),
    ("Що таке граф?", "Вузли (вершини) з'єднані ребрами. Орієнтований/неорієнтований. BFS/DFS обходи. Dijkstra для найкоротшого шляху."),
    ("Що таке SQL injection?", "Атака через вставку SQL коду в запит. Захист: prepared statements, параметризовані запити, екранування."),
    ("Що таке XSS?", "Атака через вставку скриптів у веб-сторінку. Захист: екранування HTML, Content-Security-Policy."),
    ("Що таке CSRF?", "Атака де зловмисник змушує користувача виконати дію. Захист: CSRF токени, SameSite cookies."),
    ("Що таке REST API?", "API використовує HTTP методи. GET для отримання, POST для створення, PUT для оновлення, DELETE для видалення."),
]
for q, a in cs: add(q, a)

# Shuffle
random.seed(42)
pairs = list(zip(qa[::2], qa[1::2]))
random.shuffle(pairs)
qa_flat = [x for pair in pairs for x in pair]

with open('D:/Users/ShadeFurryIT/Desktop/ShadeAI C++ Test/data/code_v2.txt', 'w', encoding='utf-8') as f:
    for line in qa_flat: f.write(line + '\n')

t = '\n'.join(qa_flat)
print('Generated %d lines (%d QA pairs), %d bytes' % (len(qa_flat), len(qa_flat)//2, len(t.encode('utf-8'))))
