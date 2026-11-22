#!/usr/bin/env python3
"""
test_generator.py — Generador de tests sintéticos para código Python.

Este módulo genera tests unitarios sintéticos para validar que el código
Python producido por el modelo (o escrito manualmente) es correcto y
cumple con expectativas básicas de comportamiento.

Incluye:
  - Generación de tests para funciones (entradas/salidas esperadas).
  - Generación de tests para clases (instanciación, métodos).
  - Tests de sintaxis (que el código parsee correctamente).
  - Tests de importación (que los imports resuelvan).
  - Ejecución de tests en un entorno aislado (subproceso).
  - Reporte de resultados en JSON y Markdown.

NOTA: Los tests se ejecutan en subprocesos separados para evitar
ejecutar código potencialmente inseguro en el proceso principal.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional


@dataclass
class TestCase:
    """Un caso de test individual."""
    name: str
    code: str  # código del test (una función de test o script)
    expected_pass: bool = True
    category: str = "unit"  # unit, syntax, import, behavior

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "category": self.category,
            "expected_pass": self.expected_pass,
        }


@dataclass
class TestResult:
    """Resultado de ejecutar un test."""
    name: str
    passed: bool
    category: str
    output: str
    error: Optional[str] = None
    elapsed_sec: float = 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "category": self.category,
            "output": self.output,
            "error": self.error,
            "elapsed_sec": self.elapsed_sec,
        }


@dataclass
class TestSuite:
    """Una suite de tests."""
    name: str
    test_cases: List[TestCase] = field(default_factory=list)
    results: List[TestResult] = field(default_factory=list)

    def add_test(self, test_case: TestCase) -> None:
        self.test_cases.append(test_case)

    def summary(self) -> dict:
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        return {
            "suite_name": self.name,
            "total_tests": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": passed / total if total > 0 else 0.0,
        }


class TestGenerator:
    """
    Generador de tests sintéticos para código Python.

    Analiza el código fuente y genera tests que verifican:
    1. Que el código es sintácticamente válido.
    2. Que las funciones definidas se pueden importar.
    3. Que las funciones producen salidas esperadas para entradas dadas.
    4. Que las clases se pueden instanciar y sus métodos funcionan.
    """

    def __init__(self) -> None:
        self.builtins: set = set(dir(__builtins__))  # type: ignore

    def generate_syntax_tests(self, code: str) -> List[TestCase]:
        """
        Genera tests que verifican que el código es sintácticamente válido.
        """
        tests: List[TestCase] = []

        test_code = f'''
import ast
import sys

code = ''' + repr(code) + '''

try:
    tree = ast.parse(code)
    print("SYNTAX_OK")
except SyntaxError as e:
    print(f"SYNTAX_FAIL: {{e.msg}} en línea {{e.lineno}}")
    sys.exit(1)
except Exception as e:
    print(f"SYNTAX_FAIL: {{e}}")
    sys.exit(1)
'''

        tests.append(TestCase(
            name="test_syntax_valid",
            code=test_code,
            category="syntax",
            expected_pass=True,
        ))

        return tests

    def generate_import_tests(self, code: str, module_name: str = "generated_module") -> List[TestCase]:
        """
        Genera tests que verifican que el código se puede importar como módulo.
        """
        tests: List[TestCase] = []

        test_code = f'''
import importlib.util
import sys
import tempfile
import os

code = ''' + repr(code) + '''

# Escribir el código a un archivo temporal
with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
    f.write(code)
    tmp_path = f.name

try:
    spec = importlib.util.spec_from_file_location("{module_name}", tmp_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    print("IMPORT_OK")
except Exception as e:
    print(f"IMPORT_FAIL: {{e}}")
    sys.exit(1)
finally:
    os.unlink(tmp_path)
'''

        tests.append(TestCase(
            name="test_module_importable",
            code=test_code,
            category="import",
            expected_pass=True,
        ))

        return tests

    def generate_function_tests(
        self,
        code: str,
        function_name: str,
        test_inputs: List[tuple],
        expected_outputs: List[object],
    ) -> List[TestCase]:
        """
        Genera tests para una función específica con entradas/salidas esperadas.

        test_inputs: lista de tuplas con argumentos posicionales.
        expected_outputs: lista de salidas esperadas (una por entrada).
        """
        tests: List[TestCase] = []

        if len(test_inputs) != len(expected_outputs):
            raise ValueError("test_inputs y expected_outputs deben tener la misma longitud")

        for i, (inp, expected) in enumerate(zip(test_inputs, expected_outputs)):
            # Serializar entrada como código Python
            args_str = ", ".join(repr(arg) for arg in inp)
            expected_repr = repr(expected)

            test_code = f'''
import importlib.util
import sys
import tempfile
import os

code = ''' + repr(code) + '''

with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
    f.write(code)
    tmp_path = f.name

try:
    spec = importlib.util.spec_from_file_location("gen_mod_{i}", tmp_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    func = getattr(module, "{function_name}")
    result = func({args_str})

    if result == {expected_repr}:
        print("TEST_OK")
    else:
        print(f"TEST_FAIL: esperado {{{expected_repr}}}, obtenido {{result!r}}")
        sys.exit(1)
except Exception as e:
    print(f"TEST_FAIL: excepcion {{e}}")
    sys.exit(1)
finally:
    os.unlink(tmp_path)
'''

            tests.append(TestCase(
                name=f"test_{function_name}_case_{i}",
                code=test_code,
                category="unit",
                expected_pass=True,
            ))

        return tests

    def generate_class_tests(
        self,
        code: str,
        class_name: str,
        init_args: List[object],
        method_calls: List[dict],
    ) -> List[TestCase]:
        """
        Genera tests para una clase.

        method_calls: lista de dicts con "method", "args", "expected".
        """
        tests: List[TestCase] = []

        args_str = ", ".join(repr(a) for a in init_args)

        # Construir llamadas a métodos como código
        calls_code = []
        for call in method_calls:
            method = call["method"]
            call_args = ", ".join(repr(a) for a in call.get("args", []))
            expected = repr(call.get("expected", None))
            calls_code.append(
                f'    result = obj.{method}({call_args})\n'
                f'    if result != {expected}:\n'
                f'        print(f"METHOD_FAIL: {{method}} esperado {{{expected}}}, obtenido {{result!r}}")\n'
                f'        sys.exit(1)\n'
            )

        calls_str = "".join(calls_code)

        test_code = f'''
import importlib.util
import sys
import tempfile
import os

code = ''' + repr(code) + '''

with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
    f.write(code)
    tmp_path = f.name

try:
    spec = importlib.util.spec_from_file_location("gen_cls_mod", tmp_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cls = getattr(module, "{class_name}")
    obj = cls({args_str})

{calls_str}
    print("CLASS_OK")
except Exception as e:
    print(f"CLASS_FAIL: excepcion {{e}}")
    sys.exit(1)
finally:
    os.unlink(tmp_path)
'''

        tests.append(TestCase(
            name=f"test_{class_name}_behavior",
            code=test_code,
            category="behavior",
            expected_pass=True,
        ))

        return tests

    def run_test(self, test_case: TestCase, timeout: float = 10.0) -> TestResult:
        """
        Ejecuta un caso de test en un subproceso separado.

        Returns
        -------
        TestResult con el resultado de la ejecución.
        """
        import time
        start = time.time()

        try:
            result = subprocess.run(
                [sys.executable, "-c", test_case.code],
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            elapsed = time.time() - start

            output = result.stdout.strip()
            error = result.stderr.strip() if result.returncode != 0 else None

            passed = (result.returncode == 0) and ("OK" in output or "PASS" in output)
            # Si se esperaba fallar y falló, también es un "pass" del test
            if not test_case.expected_pass and result.returncode != 0:
                passed = True

            return TestResult(
                name=test_case.name,
                passed=passed,
                category=test_case.category,
                output=output,
                error=error,
                elapsed_sec=round(elapsed, 4),
            )

        except subprocess.TimeoutExpired:
            elapsed = time.time() - start
            return TestResult(
                name=test_case.name,
                passed=False,
                category=test_case.category,
                output="",
                error="Timeout",
                elapsed_sec=round(elapsed, 4),
            )
        except Exception as e:
            elapsed = time.time() - start
            return TestResult(
                name=test_case.name,
                passed=False,
                category=test_case.category,
                output="",
                error=str(e),
                elapsed_sec=round(elapsed, 4),
            )

    def run_suite(self, suite: TestSuite, timeout: float = 10.0) -> TestSuite:
        """
        Ejecuta todos los tests de una suite.

        Returns
        -------
        La suite con resultados populados.
        """
        suite.results = []
        for test_case in suite.test_cases:
            result = self.run_test(test_case, timeout=timeout)
            suite.results.append(result)
        return suite

    def generate_suite_from_code(
        self,
        code: str,
        suite_name: str = "generated_suite",
    ) -> TestSuite:
        """
        Genera una suite completa de tests a partir de código Python.

        Incluye:
        - Test de sintaxis.
        - Test de importación.
        - Tests para cada función detectada (con entradas dummy).
        - Tests para cada clase detectada.
        """
        suite = TestSuite(name=suite_name)

        # Test de sintaxis
        for t in self.generate_syntax_tests(code):
            suite.add_test(t)

        # Test de importación
        for t in self.generate_import_tests(code):
            suite.add_test(t)

        # Extraer funciones y clases
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return suite  # solo sintaxis e importación

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Generar test con entrada dummy (None) y salida esperada None
                # Esto es solo una validación básica de que la función se puede llamar
                test_inputs = [()]
                expected_outputs = [None]
                try:
                    for t in self.generate_function_tests(
                        code, node.name, test_inputs, expected_outputs
                    ):
                        suite.add_test(t)
                except Exception:
                    pass  # si falla la generación, continuar

            elif isinstance(node, ast.ClassDef):
                # Test básico de instanciación sin argumentos
                try:
                    for t in self.generate_class_tests(
                        code, node.name, init_args=[], method_calls=[]
                    ):
                        suite.add_test(t)
                except Exception:
                    pass

        return suite

    def save_results(
        self,
        suite: TestSuite,
        output_path: str | Path,
        format: str = "json",
    ) -> None:
        """
        Guarda los resultados de una suite en un archivo.

        format: "json" o "markdown".
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        summary = suite.summary()
        results = [r.to_dict() for r in suite.results]

        if format == "json":
            data = {
                "suite": summary,
                "test_cases": [t.to_dict() for t in suite.test_cases],
                "results": results,
            }
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2)
        elif format == "markdown":
            lines: List[str] = []
            lines.append(f"# Resultados de Tests: {suite.name}")
            lines.append("")
            lines.append(f"Total: {summary['total_tests']} | "
                        f"Pasadas: {summary['passed']} | "
                        f"Falladas: {summary['failed']} | "
                        f"Tasa: {summary['pass_rate']:.1%}")
            lines.append("")
            lines.append("| Test | Categoría | Resultado | Salida |")
            lines.append("|------|-----------|-----------|--------|")
            for r in suite.results:
                status = "✓" if r.passed else "✗"
                lines.append(
                    f"| {r.name} | {r.category} | {status} | {r.output[:50]} |"
                )
            with open(output_path, "w") as f:
                f.write("\n".join(lines))
        else:
            raise ValueError(f"Formato no soportado: {format}")


def quick_test_code(
    code: str,
    suite_name: str = "quick_test",
) -> dict:
    """
    Función de conveniencia: genera y ejecuta una suite de tests
    para un fragmento de código Python.

    Returns
    -------
    dict con el resumen de la suite.
    """
    generator = TestGenerator()
    suite = generator.generate_suite_from_code(code, suite_name)
    suite = generator.run_suite(suite)
    return suite.summary()


__all__ = [
    "TestCase",
    "TestResult",
    "TestSuite",
    "TestGenerator",
    "quick_test_code",
]
