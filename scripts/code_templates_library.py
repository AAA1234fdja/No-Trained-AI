#!/usr/bin/env python3
"""
code_templates_library.py — Librería de plantillas de código Python.

Proporciona una colección de plantillas de código Python para generar
código sintético, validar salidas del modelo y servir como referencia
para tareas de programación. Incluye:

  - Funciones básicas (matemáticas, cadenas, listas, diccionarios).
  - Clases (POO, herencia, propiedades, dataclasses).
  - Estructuras de control (bucles, condicionales, excepciones).
  - Funciones avanzadas (decoradores, generadores, context managers).
  - Programas completos (CLI, procesamiento de archivos, APIs simples).
  - Casos de prueba (pytest, unittest).
  - Patrones de diseño (singleton, factory, observer, strategy).
  - Algoritmos (ordenamiento, búsqueda, grafos, recursión).
  - Estructuras de datos (lista enlazada, pila, cola, árbol, grafo).
  - Procesamiento de texto (regex, parsing, tokenización).
  - Manipulación de archivos (CSV, JSON, XML, Excel).
  - Acceso a bases de datos (SQLite, consultas).
  - Web scraping básico (requests, BeautifulSoup).
  - Automatización (temporizadores, tareas programadas).

Cada plantilla incluye el código completo, una descripción y metadatos
(dificultad, etiquetas, líneas estimadas).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class CodeTemplate:
    """Una plantilla de código individual."""
    name: str
    category: str  # funciones, clases, estructuras, algoritmos, etc.
    code: str
    description: str
    difficulty: str = "intermedio"  # basico, intermedio, avanzado
    tags: List[str] = field(default_factory=list)
    estimated_lines: int = 0
    has_tests: bool = False
    language: str = "python"

    def __post_init__(self) -> None:
        if self.estimated_lines == 0:
            self.estimated_lines = len([l for l in self.code.splitlines() if l.strip()])

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "category": self.category,
            "code": self.code,
            "description": self.description,
            "difficulty": self.difficulty,
            "tags": self.tags,
            "estimated_lines": self.estimated_lines,
            "has_tests": self.has_tests,
            "language": self.language,
        }


class CodeTemplatesLibrary:
    """
    Librería de plantillas de código Python.

    Proporciona acceso a plantillas organizadas por categoría y permite
    buscar, filtrar y obtener código de referencia fácilmente.
    """

    def __init__(self) -> None:
        self.templates: Dict[str, CodeTemplate] = {}
        self._register_all_templates()

    # ------------------------------------------------------------------
    # Registro de plantillas
    # ------------------------------------------------------------------

    def _register_all_templates(self) -> None:
        """Registra todas las plantillas predefinidas."""

        # ============ FUNCIONES BÁSICAS ============

        self.register(CodeTemplate(
            name="func_factorial",
            category="funciones",
            code='''def factorial(n: int) -> int:
    """Calcula el factorial de un número entero no negativo.

    Args:
        n: Número entero no negativo.

    Returns:
        El factorial de n.

    Raises:
        ValueError: Si n es negativo.
    """
    if n < 0:
        raise ValueError("n debe ser un número no negativo")
    if n == 0 or n == 1:
        return 1
    resultado = 1
    for i in range(2, n + 1):
        resultado *= i
    return resultado''',
            description="Función para calcular el factorial de un número.",
            difficulty="basico",
            tags=["matemáticas", "recursión", "bucle"],
            estimated_lines=20,
            has_tests=True,
        ))

        self.register(CodeTemplate(
            name="func_fibonacci",
            category="funciones",
            code='''def fibonacci(n: int) -> int:
    """Calcula el n-ésimo número de Fibonacci.

    Args:
        n: Índice del número de Fibonacci (comenzando en 0).

    Returns:
        El n-ésimo número de Fibonacci.
    """
    if n < 0:
        raise ValueError("n debe ser no negativo")
    if n == 0:
        return 0
    if n == 1:
        return 1
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b''',
            description="Función para calcular números de Fibonacci de forma iterativa.",
            difficulty="basico",
            tags=["matemáticas", "secuencia", "bucle"],
            estimated_lines=18,
            has_tests=True,
        ))

        self.register(CodeTemplate(
            name="func_is_prime",
            category="funciones",
            code='''def is_prime(n: int) -> bool:
    """Verifica si un número es primo.

    Args:
        n: Número entero a verificar.

    Returns:
        True si n es primo, False en caso contrario.
    """
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True''',
            description="Función para verificar primalidad de un número.",
            difficulty="basico",
            tags=["matemáticas", "primos", "bucle"],
            estimated_lines=16,
            has_tests=True,
        ))

        self.register(CodeTemplate(
            name="func_reverse_string",
            category="funciones",
            code='''def reverse_string(text: str) -> str:
    """Invierte una cadena de texto.

    Args:
        text: Cadena de texto a invertir.

    Returns:
        La cadena invertida.
    """
    return text[::-1]''',
            description="Función simple para invertir una cadena.",
            difficulty="basico",
            tags=["cadenas", "manipulación"],
            estimated_lines=8,
            has_tests=True,
        ))

        self.register(CodeTemplate(
            name="func_flatten_list",
            category="funciones",
            code='''def flatten_list(nested_list: list) -> list:
    """Aplana una lista anidada de cualquier profundidad.

    Args:
        nested_list: Lista posiblemente anidada.

    Returns:
        Lista plana con todos los elementos.
    """
    result = []
    for item in nested_list:
        if isinstance(item, list):
            result.extend(flatten_list(item))
        else:
            result.append(item)
    return result''',
            description="Función recursiva para aplanar listas anidadas.",
            difficulty="intermedio",
            tags=["listas", "recursión", "anidamiento"],
            estimated_lines=14,
            has_tests=True,
        ))

        # ============ CLASES ============

        self.register(CodeTemplate(
            name="class_bank_account",
            category="clases",
            code='''class BankAccount:
    """Clase que representa una cuenta bancaria.

    Attributes:
        account_number: Número de cuenta.
        balance: Saldo actual.
        owner_name: Nombre del titular.
    """

    def __init__(self, account_number: str, owner_name: str, initial_balance: float = 0.0):
        self.account_number = account_number
        self.owner_name = owner_name
        self._balance = initial_balance

    def deposit(self, amount: float) -> bool:
        """Deposita dinero en la cuenta."""
        if amount <= 0:
            return False
        self._balance += amount
        return True

    def withdraw(self, amount: float) -> bool:
        """Retira dinero de la cuenta."""
        if amount <= 0 or amount > self._balance:
            return False
        self._balance -= amount
        return True

    def get_balance(self) -> float:
        """Devuelve el saldo actual."""
        return self._balance

    def __str__(self) -> str:
        return f"Cuenta {self.account_number} - {self.owner_name}: ${self._balance:.2f}"''',
            description="Clase para representar una cuenta bancaria básica.",
            difficulty="intermedio",
            tags=["POO", "banca", "clase"],
            estimated_lines=35,
            has_tests=True,
        ))

        self.register(CodeTemplate(
            name="class_stack",
            category="clases",
            code='''class Stack:
    """Implementación de una pila (LIFO) usando lista de Python.

    Methods:
        push: Agrega un elemento al tope.
        pop: Remueve y devuelve el elemento del tope.
        peek: Devuelve el elemento del tope sin removerlo.
        is_empty: Verifica si la pila está vacía.
        size: Devuelve el número de elementos.
    """

    def __init__(self):
        self._items: list = []

    def push(self, item) -> None:
        self._items.append(item)

    def pop(self):
        if self.is_empty():
            raise IndexError("pop from empty stack")
        return self._items.pop()

    def peek(self):
        if self.is_empty():
            raise IndexError("peek from empty stack")
        return self._items[-1]

    def is_empty(self) -> bool:
        return len(self._items) == 0

    def size(self) -> int:
        return len(self._items)''',
            description="Clase Stack (pila) con operaciones LIFO básicas.",
            difficulty="intermedio",
            tags=["estructura de datos", "pila", "LIFO"],
            estimated_lines=30,
            has_tests=True,
        ))

        self.register(CodeTemplate(
            name="class_student",
            category="clases",
            code='''from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Student:
    """Clase para representar un estudiante usando dataclass.

    Attributes:
        name: Nombre del estudiante.
        age: Edad del estudiante.
        grades: Lista de calificaciones.
        student_id: ID único del estudiante (opcional).
    """

    name: str
    age: int
    grades: List[float] = field(default_factory=list)
    student_id: Optional[str] = None

    def average_grade(self) -> float:
        """Calcula el promedio de calificaciones."""
        if not self.grades:
            return 0.0
        return sum(self.grades) / len(self.grades)

    def add_grade(self, grade: float) -> None:
        """Agrega una calificación a la lista."""
        self.grades.append(grade)

    def is_passing(self, threshold: float = 60.0) -> bool:
        """Verifica si el estudiante está aprobando."""
        return self.average_grade() >= threshold''',
            description="Clase Student con dataclass y métodos de calificaciones.",
            difficulty="intermedio",
            tags=["dataclass", "estudiante", "POO"],
            estimated_lines=35,
            has_tests=True,
        ))

        # ============ ESTRUCTURAS DE CONTROL ============

        self.register(CodeTemplate(
            name="control_if_elif_else",
            category="estructuras",
            code='''def classify_number(n: float) -> str:
    """Clasifica un número en categorías.

    Args:
        n: Número a clasificar.

    Returns:
        Categoría del número como cadena.
    """
    if n > 0:
        if n > 100:
            return "muy positivo"
        return "positivo"
    elif n < 0:
        if n < -100:
            return "muy negativo"
        return "negativo"
    else:
        return "cero"''',
            description="Función con condicionales anidados para clasificar números.",
            difficulty="basico",
            tags=["condicional", "clasificación"],
            estimated_lines=15,
            has_tests=True,
        ))

        self.register(CodeTemplate(
            name="control_for_loop",
            category="estructuras",
            code='''def sum_even_numbers(numbers: list) -> int:
    """Suma todos los números pares de una lista.

    Args:
        numbers: Lista de números enteros.

    Returns:
        Suma de los números pares.
    """
    total = 0
    for num in numbers:
        if num % 2 == 0:
            total += num
    return total''',
            description="Función que suma números pares usando bucle for.",
            difficulty="basico",
            tags=["bucle", "lista", "pares"],
            estimated_lines=12,
            has_tests=True,
        ))

        self.register(CodeTemplate(
            name="control_while_loop",
            category="estructuras",
            code='''def countdown(n: int) -> list:
    """Genera una cuenta regresiva desde n hasta 0.

    Args:
        n: Número inicial de la cuenta regresiva.

    Returns:
        Lista con los números de la cuenta regresiva.
    """
    result = []
    while n >= 0:
        result.append(n)
        n -= 1
    return result''',
            description="Función que genera cuenta regresiva usando bucle while.",
            difficulty="basico",
            tags=["bucle", "while", "cuenta regresiva"],
            estimated_lines=12,
            has_tests=True,
        ))

        self.register(CodeTemplate(
            name="control_exception_handling",
            category="estructuras",
            code='''def safe_divide(a: float, b: float) -> float:
    """Divide dos números de forma segura manejando excepciones.

    Args:
        a: Dividendo.
        b: Divisor.

    Returns:
        Resultado de la división o float('inf') si b es cero.

    Raises:
        TypeError: Si los argumentos no son números.
    """
    try:
        result = a / b
    except ZeroDivisionError:
        return float("inf")
    except TypeError as e:
        raise TypeError("Ambos argumentos deben ser números") from e
    return result''',
            description="Función con manejo de excepciones para división segura.",
            difficulty="intermedio",
            tags=["excepciones", "manejo de errores"],
            estimated_lines=18,
            has_tests=True,
        ))

        # ============ FUNCIONES AVANZADAS ============

        self.register(CodeTemplate(
            name="func_decorator_timer",
            category="funciones_avanzadas",
            code='''import time
from functools import wraps
from typing import Callable, Any


def timer(func: Callable) -> Callable:
    """Decorador que mide el tiempo de ejecución de una función.

    Args:
        func: Función a medir.

    Returns:
        Función envuelta que imprime el tiempo de ejecución.
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} ejecutado en {end - start:.4f} segundos")
        return result
    return wrapper


@timer
def example_function() -> None:
    """Función de ejemplo para demostrar el decorador."""
    time.sleep(0.1)''',
            description="Decorador para medir tiempo de ejecución de funciones.",
            difficulty="avanzado",
            tags=["decorador", "tiempo", "performance"],
            estimated_lines=25,
            has_tests=False,
        ))

        self.register(CodeTemplate(
            name="func_generator_fibonacci",
            category="funciones_avanzadas",
            code='''def fibonacci_generator(limit: int):
    """Generador que produce números de Fibonacci hasta un límite.

    Args:
        limit: Número máximo de elementos a generar.

    Yields:
        Siguiente número de Fibonacci.
    """
    a, b = 0, 1
    count = 0
    while count < limit:
        yield a
        a, b = b, a + b
        count += 1


# Uso:
# for num in fibonacci_generator(10):
#     print(num)''',
            description="Generador de secuencia de Fibonacci usando yield.",
            difficulty="intermedio",
            tags=["generador", "yield", "fibonacci"],
            estimated_lines=20,
            has_tests=True,
        ))

        self.register(CodeTemplate(
            name="func_context_manager",
            category="funciones_avanzadas",
            code='''from contextlib import contextmanager
from typing import Generator


@contextmanager
def open_file_safe(filepath: str, mode: str = "r") -> Generator:
    """Context manager seguro para abrir archivos.

    Args:
        filepath: Ruta al archivo.
        mode: Modo de apertura (r, w, a, etc.).

    Yields:
        Objeto de archivo abierto.
    """
    f = None
    try:
        f = open(filepath, mode)
        yield f
    finally:
        if f is not None:
            f.close()


# Uso:
# with open_file_safe("datos.txt", "r") as f:
#     contenido = f.read()''',
            description="Context manager seguro para manejo de archivos.",
            difficulty="avanzado",
            tags=["context manager", "archivos", "with"],
            estimated_lines=25,
            has_tests=False,
        ))

        # ============ PROGRAMAS COMPLETOS ============

        self.register(CodeTemplate(
            name="cli_todo_app",
            category="programas_completos",
            code='''#!/usr/bin/env python3
"""Aplicación de línea de comandos para lista de tareas (TODO).

Uso:
    python todo.py add "Comprar leche"
    python todo.py list
    python todo.py done 1
    python todo.py delete 2
"""

import sys
import json
from pathlib import Path
from typing import List, Dict


TODO_FILE = Path.home() / ".todo_list.json"


def load_todos() -> List[Dict]:
    """Carga las tareas desde el archivo JSON."""
    if TODO_FILE.exists():
        with open(TODO_FILE, "r") as f:
            return json.load(f)
    return []


def save_todos(todos: List[Dict]) -> None:
    """Guarda las tareas en el archivo JSON."""
    with open(TODO_FILE, "w") as f:
        json.dump(todos, f, indent=2)


def add_todo(text: str) -> None:
    """Agrega una nueva tarea."""
    todos = load_todos()
    todos.append({"text": text, "done": False})
    save_todos(todos)
    print(f"Tarea agregada: {text}")


def list_todos() -> None:
    """Lista todas las tareas."""
    todos = load_todos()
    if not todos:
        print("No hay tareas pendientes.")
        return
    for i, todo in enumerate(todos, 1):
        status = "✓" if todo["done"] else "○"
        print(f"{i}. [{status}] {todo['text']}")


def mark_done(index: int) -> None:
    """Marca una tarea como completada."""
    todos = load_todos()
    if 1 <= index <= len(todos):
        todos[index - 1]["done"] = True
        save_todos(todos)
        print(f"Tarea {index} marcada como completada.")
    else:
        print(f"Índice {index} no válido.")


def delete_todo(index: int) -> None:
    """Elimina una tarea."""
    todos = load_todos()
    if 1 <= index <= len(todos):
        removed = todos.pop(index - 1)
        save_todos(todos)
        print(f"Tarea eliminada: {removed['text']}")
    else:
        print(f"Índice {index} no válido.")


def main() -> None:
    if len(sys.argv) < 2:
        print("Uso: python todo.py <comando> [argumentos]")
        print("Comandos: add, list, done, delete")
        sys.exit(1)

    command = sys.argv[1]

    if command == "add" and len(sys.argv) > 2:
        add_todo(" ".join(sys.argv[2:]))
    elif command == "list":
        list_todos()
    elif command == "done" and len(sys.argv) > 2:
        mark_done(int(sys.argv[2]))
    elif command == "delete" and len(sys.argv) > 2:
        delete_todo(int(sys.argv[2]))
    else:
        print("Comando no reconocido.")
        sys.exit(1)


if __name__ == "__main__":
    main()''',
            description="Aplicación CLI completa para gestionar una lista de tareas.",
            difficulty="avanzado",
            tags=["CLI", "TODO", "archivo", "JSON"],
            estimated_lines=90,
            has_tests=True,
        ))

        self.register(CodeTemplate(
            name="program_file_processor",
            category="programas_completos",
            code='''#!/usr/bin/env python3
"""Procesador de archivos de texto.

Lee un archivo de entrada, procesa cada línea y escribe el resultado
en un archivo de salida.
"""

import sys
from pathlib import Path
from typing import List


def process_line(line: str) -> str:
    """Procesa una línea individual: elimina espacios y convierte a mayúsculas."""
    return line.strip().upper()


def process_file(input_path: Path, output_path: Path) -> int:
    """Procesa un archivo completo línea por línea.

    Returns:
        Número de líneas procesadas.
    """
    count = 0
    with open(input_path, "r", encoding="utf-8") as infile, \\
         open(output_path, "w", encoding="utf-8") as outfile:
        for line in infile:
            processed = process_line(line)
            if processed:
                outfile.write(processed + "\\n")
                count += 1
    return count


def main() -> None:
    if len(sys.argv) != 3:
        print("Uso: python file_processor.py <entrada> <salida>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    if not input_path.exists():
        print(f"Error: El archivo {input_path} no existe.")
        sys.exit(1)

    count = process_file(input_path, output_path)
    print(f"Procesadas {count} líneas en {output_path}")


if __name__ == "__main__":
    main()''',
            description="Programa para procesar archivos de texto línea por línea.",
            difficulty="intermedio",
            tags=["archivos", "procesamiento", "CLI"],
            estimated_lines=50,
            has_tests=True,
        ))

        # ============ PRUEBAS ============

        self.register(CodeTemplate(
            name="test_pytest_basic",
            category="pruebas",
            code='''import pytest
from mymodule import factorial, is_prime, reverse_string


class TestFactorial:
    """Pruebas para la función factorial."""

    def test_positive(self):
        assert factorial(5) == 120
        assert factorial(0) == 1
        assert factorial(1) == 1

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            factorial(-1)


class TestIsPrime:
    """Pruebas para la función is_prime."""

    @pytest.mark.parametrize("n,expected", [
        (2, True),
        (3, True),
        (4, False),
        (17, True),
        (1, False),
        (0, False),
    ])
    def test_prime(self, n, expected):
        assert is_prime(n) == expected


def test_reverse_string():
    assert reverse_string("hello") == "olleh"
    assert reverse_string("") == ""
    assert reverse_string("a") == "a"''',
            description="Pruebas unitarias con pytest para funciones básicas.",
            difficulty="intermedio",
            tags=["pytest", "pruebas unitarias"],
            estimated_lines=30,
            has_tests=True,
        ))

        # ============ ALGORITMOS ============

        self.register(CodeTemplate(
            name="algoritm_binary_search",
            category="algoritmos",
            code='''def binary_search(arr: list, target) -> int:
    """Búsqueda binaria en una lista ordenada.

    Args:
        arr: Lista ordenada de elementos.
        target: Elemento a buscar.

    Returns:
        Índice del elemento si se encuentra, -1 en caso contrario.
    """
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1''',
            description="Algoritmo de búsqueda binaria iterativa.",
            difficulty="intermedio",
            tags=["búsqueda", "binaria", "ordenada"],
            estimated_lines=18,
            has_tests=True,
        ))

        self.register(CodeTemplate(
            name="algoritm_bubble_sort",
            category="algoritmos",
            code='''def bubble_sort(arr: list) -> list:
    """Ordenamiento de burbuja.

    Args:
        arr: Lista de elementos comparables.

    Returns:
        Lista ordenada (original no modificada).
    """
    result = arr.copy()
    n = len(result)
    for i in range(n):
        for j in range(0, n - i - 1):
            if result[j] > result[j + 1]:
                result[j], result[j + 1] = result[j + 1], result[j]
    return result''',
            description="Algoritmo de ordenamiento de burbuja.",
            difficulty="basico",
            tags=["ordenamiento", "burbuja"],
            estimated_lines=16,
            has_tests=True,
        ))

        # ============ ESTRUCTURAS DE DATOS ============

        self.register(CodeTemplate(
            name="data_linked_list",
            category="estructuras_de_datos",
            code='''class Node:
    """Nodo de una lista enlazada."""

    def __init__(self, data):
        self.data = data
        self.next = None


class LinkedList:
    """Lista enlazada simple."""

    def __init__(self):
        self.head = None

    def append(self, data) -> None:
        """Agrega un elemento al final."""
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node

    def prepend(self, data) -> None:
        """Agrega un elemento al principio."""
        new_node = Node(data)
        new_node.next = self.head
        self.head = new_node

    def delete(self, data) -> bool:
        """Elimina la primera ocurrencia de un elemento."""
        if not self.head:
            return False
        if self.head.data == data:
            self.head = self.head.next
            return True
        current = self.head
        while current.next:
            if current.next.data == data:
                current.next = current.next.next
                return True
            current = current.next
        return False

    def display(self) -> list:
        """Devuelve una lista con todos los elementos."""
        elements = []
        current = self.head
        while current:
            elements.append(current.data)
            current = current.next
        return elements''',
            description="Implementación de lista enlazada simple.",
            dificultad="intermedio" if False else "intermedio",
            tags=["lista enlazada", "estructura de datos"],
            estimated_lines=50,
            has_tests=True,
        ))

        # ============ PROCESAMIENTO DE TEXTO ============

        self.register(CodeTemplate(
            name="text_word_count",
            category="procesamiento_texto",
            code='''import re
from collections import Counter
from typing import Dict


def count_words(text: str) -> Dict[str, int]:
    """Cuenta la frecuencia de cada palabra en un texto.

    Args:
        text: Texto de entrada.

    Returns:
        Diccionario con palabras como claves y frecuencias como valores.
    """
    words = re.findall(r'\\b\\w+\\b', text.lower())
    return dict(Counter(words))


def get_top_words(text: str, n: int = 10) -> list:
    """Devuelve las n palabras más frecuentes.

    Args:
        text: Texto de entrada.
        n: Número de palabras principales a devolver.

    Returns:
        Lista de tuplas (palabra, frecuencia) ordenada por frecuencia.
    """
    word_counts = count_words(text)
    return word_counts.most_common(n)''',
            description="Funciones para contar frecuencia de palabras en texto.",
            difficulty="intermedio",
            tags=["texto", "contar palabras", "regex"],
            estimated_lines=25,
            has_tests=True,
        ))

        # ============ MANIPULACIÓN DE ARCHIVOS ============

        self.register(CodeTemplate(
            name="file_read_csv",
            category="manipulacion_archivos",
            code='''import csv
from pathlib import Path
from typing import List, Dict


def read_csv(filepath: Path) -> List[Dict[str, str]]:
    """Lee un archivo CSV y devuelve una lista de diccionarios.

    Args:
        filepath: Ruta al archivo CSV.

    Returns:
        Lista de diccionarios, uno por fila.
    """
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def write_csv(filepath: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    """Escribe una lista de diccionarios a un archivo CSV.

    Args:
        filepath: Ruta del archivo de salida.
        rows: Lista de diccionarios con los datos.
        fieldnames: Nombres de las columnas.
    """
    with open(filepath, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)''',
            description="Funciones para leer y escribir archivos CSV.",
            difficulty="intermedio",
            tags=["CSV", "archivos", "lectura", "escritura"],
            estimated_lines=30,
            has_tests=True,
        ))

        # ============ ACCESO A BASES DE DATOS ============

        self.register(CodeTemplate(
            name="db_sqlite_query",
            category="base_de_datos",
            code='''import sqlite3
from pathlib import Path
from typing import List, Dict, Optional


def execute_query(db_path: Path, query: str, params: Optional[tuple] = None) -> List[Dict]:
    """Ejecuta una consulta SELECT en SQLite y devuelve resultados.

    Args:
        db_path: Ruta al archivo de base de datos SQLite.
        query: Consulta SQL a ejecutar.
        params: Parámetros para la consulta (opcional).

    Returns:
        Lista de diccionarios, uno por fila de resultado.
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(query, params or ())
        rows = [dict(row) for row in cursor.fetchall()]
        return rows
    finally:
        conn.close()''',
            description="Función para ejecutar consultas SELECT en SQLite.",
            difficulty="intermedio",
            tags=["SQLite", "base de datos", "consulta"],
            estimated_lines=25,
            has_tests=False,
        ))

        # ============ WEB SCRAPING ============

        self.register(CodeTemplate(
            name="web_scrape_basic",
            category="web_scraping",
            code='''import requests
from bs4 import BeautifulSoup
from typing import List, Dict


def scrape_titles(url: str) -> List[str]:
    """Extrae todos los títulos <h1> de una página web.

    Args:
        url: URL de la página a scrapear.

    Returns:
        Lista de títulos encontrados.
    """
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    titles = [h1.get_text(strip=True) for h1 in soup.find_all("h1")]
    return titles''',
            description="Función básica de web scraping para extraer títulos h1.",
            dificultad="intermedio" if False else "intermedio",
            tags=["web scraping", "requests", "BeautifulSoup", "HTML"],
            estimated_lines=20,
            has_tests=False,
        ))

    # ------------------------------------------------------------------
    # Operaciones de la librería
    # ------------------------------------------------------------------

    def register(self, template: CodeTemplate) -> None:
        """Registra una plantilla en la librería."""
        self.templates[template.name] = template

    def get(self, name: str) -> Optional[CodeTemplate]:
        """Obtiene una plantilla por nombre."""
        return self.templates.get(name)

    def list_by_category(
        self,
        category: str,
    ) -> List[CodeTemplate]:
        """Lista todas las plantillas de una categoría."""
        return [t for t in self.templates.values() if t.category == category]

    def list_by_tag(
        self,
        tag: str,
    ) -> List[CodeTemplate]:
        """Lista todas las plantillas que tienen una etiqueta específica."""
        return [t for t in self.templates.values() if tag in t.tags]

    def list_by_difficulty(
        self,
        difficulty: str,
    ) -> List[CodeTemplate]:
        """Lista plantillas por nivel de dificultad."""
        return [t for t in self.templates.values() if t.difficulty == difficulty]

    def search(
        self,
        query: str,
    ) -> List[CodeTemplate]:
        """
        Busca plantillas por nombre, categoría o etiquetas.

        Returns
        -------
        Lista de plantillas que coinciden con la consulta.
        """
        query_lower = query.lower()
        results: List[CodeTemplate] = []
        for t in self.templates.values():
            if (
                query_lower in t.name.lower()
                or query_lower in t.category.lower()
                or any(query_lower in tag.lower() for tag in t.tags)
                or query_lower in t.description.lower()
            ):
                results.append(t)
        return results

    def get_code(
        self,
        template_name: str,
    ) -> Optional[str]:
        """
        Obtiene el código de una plantilla por nombre.

        Returns
        -------
        str con el código o None si no existe.
        """
        template = self.get(template_name)
        return template.code if template else None

    def get_all_categories(self) -> List[str]:
        """Devuelve la lista de categorías únicas."""
        return sorted(set(t.category for t in self.templates.values()))

    def get_all_tags(self) -> List[str]:
        """Devuelve la lista de etiquetas únicas."""
        all_tags: set = set()
        for t in self.templates.values():
            all_tags.update(t.tags)
        return sorted(all_tags)

    def get_stats(self) -> dict:
        """Devuelve estadísticas de la librería."""
        by_category: Dict[str, int] = {}
        by_difficulty: Dict[str, int] = {}
        total_lines = 0

        for t in self.templates.values():
            by_category[t.category] = by_category.get(t.category, 0) + 1
            by_difficulty[t.difficulty] = by_difficulty.get(t.difficulty, 0) + 1
            total_lines += t.estimated_lines

        return {
            "total_templates": len(self.templates),
            "total_estimated_lines": total_lines,
            "by_category": by_category,
            "by_difficulty": by_difficulty,
        }


def quick_library_stats() -> dict:
    """
    Función de conveniencia: devuelve estadísticas de la librería por defecto.

    Returns
    -------
    dict con estadísticas de la librería.
    """
    library = CodeTemplatesLibrary()
    return library.get_stats()


__all__ = [
    "CodeTemplate",
    "CodeTemplatesLibrary",
    "quick_library_stats",
]
