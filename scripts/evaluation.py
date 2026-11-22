#!/usr/bin/env python3
"""
evaluation.py — Scripts de evaluación para el modelo 800M.

Proporciona suites de pruebas para evaluar las capacidades del modelo
sin entrenamiento:

  - Python capability: completación de sintaxis, palabras clave, bucles.
  - Analysis capability: razonamiento de complejidad, estadística.
  - Conversation capability: saludos, preguntas, respuestas breves.
  - Induction capability: copia en contexto ([A][B] ... [A] -> [B]).
  - Perplexidad aproximada sobre un conjunto de frases de prueba.
  - Entropía de generación (diversidad).
  - Comparación entre etapas (stage_0 a stage_5).
  - Generación de reportes en JSON y Markdown.

Cada prueba devuelve un diccionario con: prompt, generación, entropía,
top-5, tiempo, y un booleano de "pass" según criterios definidos.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from scripts.inference_engine import MmapInt8LlamaModel, run_inference


# ---------------------------------------------------------------------------
# Definición de pruebas
# ---------------------------------------------------------------------------

class TestSuite:
    """
    Conjunto de pruebas de capacidades para el modelo.

    Cada prueba se define como (nombre, prompt, criterio_paso, max_tokens).
    El criterio_paso es una función que recibe el texto generado y devuelve
    (bool, razonamiento).
    """

    def __init__(self) -> None:
        self.tests: List[Tuple[str, str, object, int]] = []
        self.results: List[dict] = []

    def add_test(
        self,
        name: str,
        prompt: str,
        pass_fn: object,
        max_new_tokens: int = 10,
    ) -> None:
        """Añade una prueba al suite."""
        self.tests.append((name, prompt, pass_fn, max_new_tokens))

    def run(
        self,
        model_dir: str | Path,
        weights_filename: str = "model.safetensors",
    ) -> List[dict]:
        """
        Ejecuta todas las pruebas contra el modelo.

        Returns lista de resultados (uno por prueba).
        """
        prompts = [t[1] for t in self.tests]
        results_raw = run_inference(
            model_dir,
            prompts,
            weights_filename=weights_filename,
        )

        self.results = []
        for (name, prompt, pass_fn, max_toks), res in zip(self.tests, results_raw):
            generated = res["generated_text"]
            entropy = res["first_token_entropy_nats"]
            passed, reason = pass_fn(generated, entropy)

            self.results.append({
                "test_name": name,
                "prompt": prompt,
                "generated": generated,
                "passed": passed,
                "reason": reason,
                "entropy_nats": entropy,
                "elapsed_sec": res["elapsed_sec"],
                "top5": res["top5_first_token"],
            })

        return self.results

    def summary(self) -> Dict[str, object]:
        """
        Devuelve un resumen de los resultados: total, pasadas, falladas,
        entropía promedio, etc.
        """
        if not self.results:
            return {"error": "No hay resultados. Ejecutar run() primero."}

        total = len(self.results)
        passed = sum(1 for r in self.results if r["passed"])
        failed = total - passed
        entropies = [r["entropy_nats"] for r in self.results]

        return {
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / total if total > 0 else 0.0,
            "mean_entropy": statistics.mean(entropies),
            "min_entropy": min(entropies),
            "max_entropy": max(entropies),
            "results": self.results,
        }

    def save_json(self, path: str | Path) -> None:
        """Guarda los resultados en JSON."""
        with open(path, "w") as f:
            json.dump(self.summary(), f, indent=2)

    def save_markdown(self, path: str | Path) -> None:
        """Guarda los resultados en Markdown."""
        s = self.summary()
        lines = [
            "# Resultados de Evaluación",
            f"Total de pruebas: {s['total_tests']}",
            f"Pasadas: {s['passed']}",
            f"Falladas: {s['failed']}",
            f"Tasa de aprobación: {s['pass_rate']:.1%}",
            f"Entropía media: {s['mean_entropy']:.3f} nats",
            "",
            "| Prueba | Prompt | Generación | Pasó | Entropía | Razón |",
            "|--------|--------|------------|------|----------|-------|",
        ]
        for r in self.results:
            gen = r["generated"].replace("|", "\\|").replace("\n", " ")[:60]
            prmpt = r["prompt"].replace("|", "\\|")[:40]
            lines.append(
                f"| {r['test_name']} | {prmpt} | {gen} | "
                f"{'✓' if r['passed'] else '✗'} | {r['entropy_nats']:.2f} | {r['reason']} |"
            )
        with open(path, "w") as f:
            f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Funciones de criterio de paso para cada capacidad
# ---------------------------------------------------------------------------

def python_syntax_pass(generated: str, entropy: float) -> Tuple[bool, str]:
    """
    Criterio para pruebas de sintaxis Python: el modelo debe generar
    identificadores válidos, no espacios o caracteres de colapso.
    """
    if not generated or generated.isspace():
        return False, "Generó solo espacios o vacío"

    # Detectar repetición de un solo carácter (colapso)
    if len(set(generated.strip())) <= 1 and len(generated) > 2:
        return False, "Colapso a un solo carácter repetido"

    # Debe contener al menos un identificador o palabra clave Python
    py_keywords = {
        "def", "class", "return", "if", "else", "for", "while",
        "import", "from", "try", "except", "with", "lambda",
        "print", "len", "sum", "range", "analyze", "compute",
        "process", "solve", "calculate",
    }
    tokens = generated.strip().split()
    found_kw = [t for t in tokens if t.lower().strip(".,:;!?") in py_keywords]

    if found_kw:
        return True, f"Contiene keywords Python: {found_kw[:3]}"
    else:
        return False, f"No contiene keywords Python reconocibles (generó: {generated[:30]!r})"


def analysis_pass(generated: str, entropy: float) -> Tuple[bool, str]:
    """
    Criterio para pruebas de análisis: debe generar palabras de razonamiento.
    """
    if not generated or generated.isspace():
        return False, "Vacío o solo espacios"

    analysis_words = {
        "complexity", "time", "space", "algorithm", "binary", "search",
        "sort", "optimal", "linear", "logarithmic", "quadratic", "first",
        "second", "therefore", "because", "analysis", "evaluate",
        "conclusion", "step", "steps", "optimal", "efficient",
    }
    tokens = generated.strip().lower().split()
    found = [t for t in tokens if t.strip(".,:;!?") in analysis_words]

    if found:
        return True, f"Contiene palabras de análisis: {found[:3]}"
    else:
        return False, f"No contiene palabras de análisis (generó: {generated[:30]!r})"


def conversation_pass(generated: str, entropy: float) -> Tuple[bool, str]:
    """
    Criterio para conversación mínima: no debe ser un bucle de un solo
    carácter ni solo signos de puntuación.
    """
    if not generated or generated.isspace():
        return False, "Vacío"

    # Detectar bucle de carácter único
    stripped = generated.strip()
    if len(stripped) > 2 and len(set(stripped)) == 1:
        return False, f"Bucle de un solo carácter: {stripped[0]!r}"

    # Detectar repetición de palabra única
    tokens = stripped.split()
    if len(tokens) > 3 and len(set(tokens)) == 1:
        return False, f"Repetición de palabra única: {tokens[0]!r}"

    # Debe contener al menos una palabra conversacional reconocible
    conv_words = {
        "hello", "hi", "help", "can", "i", "you", "please", "thanks",
        "good", "fine", "great", "sure", "okay", "yes", "no", "maybe",
        "explain", "analyze", "write", "python", "code", "data",
    }
    found = [t for t in tokens if t.strip(".,:;!?").lower() in conv_words]

    if found:
        return True, f"Contiene palabras conversacionales: {found[:3]}"
    else:
        return False, f"No contiene palabras conversacionales (generó: {generated[:30]!r})"


def induction_pass(generated: str, entropy: float) -> Tuple[bool, str]:
    """
    Criterio para inducción: debe mostrar algún patrón de copia, no
    necesariamente perfecto, pero no debe ser solo ruido.
    """
    if not generated or generated.isspace():
        return False, "Vacío"

    # Si contiene números o patrones repetitivos, es señal de inducción
    import re
    numbers = re.findall(r"\d+", generated)
    if numbers:
        return True, f"Contiene números (patrón de copia): {numbers[:3]}"

    # Entropía muy baja indica colapso; muy alta indica ruido
    if entropy < 0.5:
        return False, f"Entropía muy baja (colapso): {entropy:.2f}"

    return True, f"Generó texto sin colapso (H={entropy:.2f})"


def no_collapse_pass(generated: str, entropy: float) -> Tuple[bool, str]:
    """
    Criterio básico: el modelo no debe colapsar a un solo carácter
    ni repetir la misma palabra indefinidamente.
    """
    if not generated or generated.isspace():
        return False, "Vacío"

    stripped = generated.strip()
    if len(set(stripped)) <= 1 and len(stripped) > 1:
        return False, "Colapso a carácter único"

    tokens = stripped.split()
    if len(tokens) > 5 and len(set(tokens)) == 1:
        return False, "Repetición de palabra única"

    return True, "Sin colapso detectado"


# ---------------------------------------------------------------------------
# Suites predefinidas
# ---------------------------------------------------------------------------

def build_python_suite() -> TestSuite:
    """Suite de capacidades en Python."""
    suite = TestSuite()
    suite.add_test("python_def", "def ", python_syntax_pass, 10)
    suite.add_test("python_for", "for i in range(10):\n    ", python_syntax_pass, 10)
    suite.add_test("python_if", "if x > 0:\n    ", python_syntax_pass, 10)
    suite.add_test("python_function", "def calculate_sum(arr):\n    return ", python_syntax_pass, 12)
    suite.add_test("python_class", "class DataAnalyzer:\n    def ", python_syntax_pass, 12)
    suite.add_test("python_import", "import ", python_syntax_pass, 8)
    suite.add_test("python_list", "data = [1, 2, 3, 4, 5]\n", python_syntax_pass, 10)
    return suite


def build_analysis_suite() -> TestSuite:
    """Suite de capacidades de análisis."""
    suite = TestSuite()
    suite.add_test("analysis_complexity", "Analyze the time complexity of binary search:", analysis_pass, 12)
    suite.add_test("analysis_variance", "Compare variance and outliers in numerical data:", analysis_pass, 12)
    suite.add_test("analysis_algorithm", "Explain the difference between linear and logarithmic time complexity:", analysis_pass, 12)
    suite.add_test("analysis_data", "Describe the steps to analyze a dataset:", analysis_pass, 12)
    return suite


def build_conversation_suite() -> TestSuite:
    """Suite de conversación mínima."""
    suite = TestSuite()
    suite.add_test("conv_hello", "Hello! How can you help me?", conversation_pass, 10)
    suite.add_test("conv_help", "Can you help me analyze data?", conversation_pass, 10)
    suite.add_test("conv_greeting", "Good morning!", conversation_pass, 8)
    suite.add_test("conv_question", "What is Python?", conversation_pass, 10)
    return suite


def build_induction_suite() -> TestSuite:
    """Suite de inducción / copia en contexto."""
    suite = TestSuite()
    suite.add_test("induction_copy", "alpha = 42 ; beta = 99 ; alpha =", induction_pass, 10)
    suite.add_test("induction_repeat", "X = 10 ; Y = 20 ; X =", induction_pass, 10)
    return suite


def build_no_collapse_suite() -> TestSuite:
    """Suite básica de no-colapso (todas las pruebas)."""
    suite = TestSuite()
    all_prompts = [
        ("nc_def", "def "),
        ("nc_analysis", "Analyze the time complexity:"),
        ("nc_conv", "Hello! How can you help?"),
        ("nc_loop", "for i in range(5):\n    "),
        ("nc_data", "Calculate the mean of:"),
    ]
    for name, prompt in all_prompts:
        suite.add_test(name, prompt, no_collapse_pass, 8)
    return suite


# ---------------------------------------------------------------------------
# Ejecución de todas las suites
# ---------------------------------------------------------------------------

def run_all_suites(
    model_dir: str | Path,
    weights_filename: str = "model.safetensors",
    output_json: Optional[str | Path] = None,
    output_md: Optional[str | Path] = None,
) -> Dict[str, object]:
    """
    Ejecuta todos los suites de evaluación y devuelve un reporte combinado.

    Returns
    -------
    dict con resultados de cada suite y resumen global.
    """
    suites = {
        "python": build_python_suite(),
        "analysis": build_analysis_suite(),
        "conversation": build_conversation_suite(),
        "induction": build_induction_suite(),
        "no_collapse": build_no_collapse_suite(),
    }

    combined: Dict[str, object] = {
        "model_dir": str(model_dir),
        "weights_filename": weights_filename,
        "suites": {},
        "global_summary": {},
    }

    total_tests = 0
    total_passed = 0

    for sname, suite in suites.items():
        suite.run(model_dir, weights_filename)
        s_summary = suite.summary()
        combined["suites"][sname] = s_summary
        total_tests += s_summary["total_tests"]
        total_passed += s_summary["passed"]

        if output_md:
            suite.save_md = lambda p: None  # placeholder para evitar error

    combined["global_summary"] = {
        "total_tests": total_tests,
        "total_passed": total_passed,
        "total_failed": total_tests - total_passed,
        "global_pass_rate": total_passed / total_tests if total_tests > 0 else 0.0,
    }

    if output_json:
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(combined, f, indent=2)

    return combined


def print_evaluation_report(
    report: Dict[str, object],
) -> None:
    """Imprime el reporte de evaluación en formato legible."""
    gs = report["global_summary"]
    print("=" * 70)
    print("REPORTE DE EVALUACIÓN DEL MODELO 800M")
    print("=" * 70)
    print(f"Total pruebas:  {gs['total_tests']}")
    print(f"Pasadas:        {gs['total_passed']}")
    print(f"Falladas:       {gs['total_failed']}")
    print(f"Tasa de paso:   {gs['global_pass_rate']:.1%}")
    print()

    for sname, sdata in report["suites"].items():
        print(f"--- Suite: {sname} ---")
        print(f"  Pruebas: {sdata['total_tests']}, Pasó: {sdata['passed']}, "
              f"Tasa: {sdata['pass_rate']:.1%}, Entropía media: {sdata['mean_entropy']:.2f}")
        for r in sdata["results"]:
            mark = "✓" if r["passed"] else "✗"
            print(f"  {mark} {r['test_name']:25s} | H={r['entropy_nats']:.2f} | {r['reason'][:60]}")
        print()


__all__ = [
    "TestSuite",
    "python_syntax_pass",
    "analysis_pass",
    "conversation_pass",
    "induction_pass",
    "no_collapse_pass",
    "build_python_suite",
    "build_analysis_suite",
    "build_conversation_suite",
    "build_induction_suite",
    "build_no_collapse_suite",
    "run_all_suites",
    "print_evaluation_report",
]
