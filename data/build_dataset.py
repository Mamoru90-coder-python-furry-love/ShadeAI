import io, random
random.seed(42)

def read(fn, max_chars=None):
    try:
        t = io.open(fn, encoding="utf-8-sig", errors="replace").read()
    except Exception:
        try:
            t = io.open(fn, encoding="utf-8", errors="replace").read()
        except Exception:
            return ""
    if max_chars: t = t[:max_chars]
    return t

parts = []
# 1) UK instruction data (UAlpaca) - core Q&A
t = read("train_mega2.txt")
print("train_mega2:", len(t), "chars")
parts.append(t)

# 2) English fiction/stories - for English
t = read("tinystories.txt", max_chars=6000000)
print("tinystories (6M):", len(t), "chars")
parts.append(t)

# 3) Ukrainian fiction - for fluent Ukrainian
t = read("ukr_fiction.txt", max_chars=3000000)
print("ukr_fiction (3M):", len(t), "chars")
parts.append(t)

# 4) code v2 (JS/C++ etc)
t = read("code_v2.txt")
print("code_v2:", len(t), "chars")
parts.append(t)

# 5) oasst uk
t = read("oasst_uk.txt")
print("oasst_uk:", len(t), "chars")
parts.append(t)

# Build Russian text programmatically (transliterated instruction-style examples)
ru_examples = [
    "USER: Напиши функцию на Python для суммы двух чисел\nAI: def add(a, b):\n    return a + b",
    "USER: Как отсортировать список в Python?\nAI: Используйте метод sort() или функцию sorted().\nНапример: my_list = [3, 1, 2]; my_list.sort()",
    "USER: Объясни что такое переменная\nAI: Переменная - это именованная область памяти для хранения данных.\nВ Python: x = 5",
    "USER: Напиши цикл на C++\nAI: for (int i = 0; i < 10; i++) {\n    std::cout << i << std::endl;\n}",
    "USER: Как сделать if в Python?\nAI: if x > 0:\n    print(\"положительное\")\nelse:\n    print(\"отрицательное\")",
    "USER: Что такое функция?\nAI: Функция - это блок кода, который выполняет определённую задачу.\nОна принимает аргументы и возвращает результат.",
    "USER: Напиши hello world на C#\nAI: using System;\nclass Program {\n    static void Main() {\n        Console.WriteLine(\"Hello, world!\");\n    }\n}",
    "USER: Как объявить массив в Java?\nAI: int[] arr = new int[5];\nили\nint[] arr = {1, 2, 3, 4, 5};",
    "USER: Что такое цикл?\nAI: Цикл - это конструкция, которая повторяет блок кода несколько раз.\nwhile и for - основные виды циклов.",
    "USER: Напиши класс на C++\nAI: class Dog {\npublic:\n    std::string name;\n    void bark() { std::cout << \"Woof!\" << std::endl; }\n};",
]
ru_text = "\n\n".join(ru_examples) * 40  # amplify
print("ru_gen:", len(ru_text), "chars")
parts.append(ru_text)

# Concatenate, strip BOMs
out = "\n\n".join(parts)
out = out.replace("\ufeff", "")

with io.open("train_full3.txt", "w", encoding="utf-8") as f:
    f.write(out)

print("TOTAL:", len(out), "chars -> train_full3.txt")
