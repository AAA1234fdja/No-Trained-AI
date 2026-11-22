#!/usr/bin/env python3
"""
cli_tools.py — Herramientas de línea de comandos para el experimento AI 800M.

Proporciona una interfaz CLI para:
  - Validar el modelo (model_validator).
  - Ejecutar benchmarks (benchmark).
  - Ejecutar evaluaciones (evaluation).
  - Generar texto (inference_engine).
  - Reensamblar las 10 partes del .safetensors.
  - Sintetizar pesos (synthesize_weights / run_stage5).

Cada comando es una función independiente que se puede llamar desde
línea de comandos o importar como módulo.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

# Importar módulos del proyecto
from scripts.logging_utils import setup_logging, timer, ProgressBar
from scripts.config_manager import ExperimentConfig, model_size_config
from scripts.model_validator import quick_validate, ModelValidator
from scripts.benchmark import quick_benchmark, ModelBenchmark
from scripts.evaluation import (
    run_all_suites,
    print_evaluation_report,
    build_python_suite,
    build_analysis_suite,
    build_conversation_suite,
    build_induction_suite,
    build_no_collapse_suite,
)
from scripts.inference_engine import MmapInt8LlamaModel, run_inference
from scripts.weight_editor import WeightEditor, quick_edit_safetensors
from scripts.weight_analyzer import WeightAnalyzer, quick_analyze
from scripts.model_builder import (
    build_tensor_spec,
    verify_tensor_spec_consistency,
    write_config,
    validate_config,
    estimate_disk_size_mb,
    count_total_params,
)


logger = setup_logging(level=20)  # INFO


# ---------------------------------------------------------------------------
# Comandos individuales
# ---------------------------------------------------------------------------

def cmd_validate(args: argparse.Namespace) -> int:
    """
    Valida el modelo .safetensors y su configuración.
    """
    model_dir = Path(args.model_dir)
    weights_file = args.weights_file or "model.safetensors"

    logger.info(f"Validando modelo en {model_dir}")

    with timer("validación completa", logger):
        report = quick_validate(model_dir, include_dynamic=args.dynamic)

    logger.info(
        f"Resultado: {report['passed']}/{report['total_checks']} checks pasados, "
        f"todos pasados: {report['all_passed']}"
    )

    if args.output:
        validator = ModelValidator(model_dir, weights_file)
        validator.save_report(args.output, report)
        logger.info(f"Reporte guardado en {args.output}")

    if not report["all_passed"]:
        return 1
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Ejecuta benchmarks de rendimiento."""
    model_dir = Path(args.model_dir)
    parts_dir = Path(args.parts_dir) if args.parts_dir else None

    logger.info(f"Ejecutando benchmarks en {model_dir}")

    with timer("benchmarks", logger):
        report = quick_benchmark(model_dir, parts_dir=parts_dir)

    logger.info("Resultados de benchmarks:")
    for r in report["benchmark_results"]:
        logger.info(
            f"  {r['name']}: {r['value']:.4f} {r['unit']}"
        )

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Reporte guardado en {args.output}")

    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Ejecuta suites de evaluación del modelo."""
    model_dir = Path(args.model_dir)
    weights_file = args.weights_file or "model.safetensors"

    logger.info(f"Ejecutando evaluación en {model_dir}")

    suites_to_run = args.suites or [
        "python", "analysis", "conversation", "induction", "no_collapse"
    ]

    suite_map = {
        "python": build_python_suite(),
        "analysis": build_analysis_suite(),
        "conversation": build_conversation_suite(),
        "induction": build_induction_suite(),
        "no_collapse": build_no_collapse_suite(),
    }

    combined: dict = {
        "model_dir": str(model_dir),
        "weights_file": weights_file,
        "suites": {},
        "global_summary": {},
    }

    total_tests = 0
    total_passed = 0

    for sname in suites_to_run:
        if sname not in suite_map:
            logger.warning(f"Suite desconocida: {sname}")
            continue

        suite = suite_map[sname]
        logger.info(f"Ejecutando suite: {sname}")
        suite.run(model_dir, weights_file)
        s_summary = suite.summary()
        combined["suites"][sname] = s_summary
        total_tests += s_summary["total_tests"]
        total_passed += s_summary["passed"]

    combined["global_summary"] = {
        "total_tests": total_tests,
        "total_passed": total_passed,
        "total_failed": total_tests - total_passed,
        "global_pass_rate": total_passed / total_tests if total_tests > 0 else 0.0,
    }

    logger.info(
        f"Resultado global: {total_passed}/{total_tests} pruebas pasadas "
        f"({combined['global_summary']['global_pass_rate']:.1%})"
    )

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(combined, f, indent=2)
        logger.info(f"Reporte guardado en {args.output}")

    if args.markdown:
        Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
        # Usar la última suite para generar markdown (simplificado)
        last_suite = list(suite_map.values())[0]
        last_suite.save_markdown(args.markdown)
        logger.info(f"Reporte Markdown guardado en {args.markdown}")

    return 0 if combined["global_summary"]["global_pass_rate"] >= 0.5 else 1


def cmd_generate(args: argparse.Namespace) -> int:
    """Genera texto usando el modelo."""
    model_dir = Path(args.model_dir)
    weights_file = args.weights_file or "model.safetensors"

    prompts = args.prompt or [
        "def fibonacci(n):",
        "Hello! How can you help me?",
        "Analyze the time complexity of binary search:",
    ]

    logger.info(f"Generando texto para {len(prompts)} prompts")

    results = run_inference(
        model_dir,
        prompts,
        weights_filename=weights_file,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
    )

    for res in results:
        print(f"Prompt:  {res['prompt']!r}")
        print(f"Gen:     {res['generated_text']!r}")
        print(f"Entropy: {res['first_token_entropy_nats']:.3f} nats")
        print(f"Time:    {res['elapsed_sec']}s")
        print(f"TPS:     {res['tokens_per_second']}")
        print()

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Resultados guardados en {args.output}")

    return 0


def cmd_reassemble(args: argparse.Namespace) -> int:
    """Reensambla las 10 partes del .safetensors en un archivo completo."""
    parts_dir = Path(args.parts_dir)
    output_file = Path(args.output) if args.output else parts_dir / "model.safetensors"

    part_files = sorted(parts_dir.glob("model.safetensors.part.*"))

    if not part_files:
        logger.error(f"No se encontraron partes en {parts_dir}")
        return 1

    logger.info(f"Reensamblando {len(part_files)} partes en {output_file}")

    with timer("reensamblaje", logger):
        total_size = 0
        with open(output_file, "wb") as out:
            with ProgressBar(len(part_files), prefix="Partes") as pb:
                for i, part_file in enumerate(part_files):
                    with open(part_file, "rb") as inp:
                        out.write(inp.read())
                    total_size += part_file.stat().st_size
                    pb.update(i + 1)

    size_mb = total_size / (1024 * 1024)
    logger.info(
        f"Reensamblado completo: {output_file} ({size_mb:.2f} MB)"
    )

    # Validar que el archivo reensamblado sea un .safetensors válido
    from scripts.model_validator import quick_validate
    report = quick_validate(parts_dir, include_dynamic=False)
    logger.info(
        f"Validación post-reensamblaje: {report['passed']}/{report['total_checks']} checks"
    )

    return 0 if report["all_passed"] else 1


def cmd_synthesize(args: argparse.Namespace) -> int:
    """
    Ejecuta la síntesis de pesos (etapas 0-5).

    Requiere que los scripts synthesize_weights.py y run_stage5.py estén
    disponibles en el directorio scripts/.
    """
    output_dir = Path(args.output_dir) if args.output_dir else Path("/home/user/ai_experiment")

    logger.info(f"Iniciando síntesis de pesos en {output_dir}")

    try:
        import subprocess
        # Ejecutar build_tokenizer.py
        logger.info("Ejecutando build_tokenizer.py")
        subprocess.run(
            [sys.executable, "scripts/build_tokenizer.py"],
            check=True,
            cwd=str(Path.cwd()),
        )

        # Ejecutar synthesize_weights.py (etapas 0-4)
        logger.info("Ejecutando synthesize_weights.py (etapas 0-4)")
        subprocess.run(
            [sys.executable, "scripts/synthesize_weights.py"],
            check=True,
            cwd=str(Path.cwd()),
        )

        # Ejecutar run_stage5.py (etapa 5)
        logger.info("Ejecutando run_stage5.py (etapa 5)")
        subprocess.run(
            [sys.executable, "scripts/run_stage5.py"],
            check=True,
            cwd=str(Path.cwd()),
        )

        logger.info("Síntesis completada exitosamente")
        return 0

    except subprocess.CalledProcessError as e:
        logger.error(f"Error en la síntesis: {e}")
        return 1
    except FileNotFoundError as e:
        logger.error(f"Script no encontrado: {e}")
        return 1


def cmd_config(args: argparse.Namespace) -> int:
    """Gestiona configuraciones del experimento."""
    if args.action == "show":
        config = ExperimentConfig()
        print(json.dumps(config.to_dict(), indent=2))
    elif args.action == "validate":
        config_path = Path(args.config_file)
        if not config_path.exists():
            logger.error(f"Archivo no encontrado: {config_path}")
            return 1
        config = ExperimentConfig.load_json(config_path)
        required = [
            "model.vocab_size", "model.hidden_size", "model.num_hidden_layers",
            "model.num_attention_heads", "model.num_key_value_heads",
            "model.head_dim", "model.intermediate_size",
        ]
        missing = ExperimentConfig.validate_config_keys(config.to_dict(), required)
        if missing:
            logger.error(f"Claves faltantes: {missing}")
            return 1
        logger.info("Configuración válida")
    elif args.action == "size":
        try:
            cfg = model_size_config(args.size_name)
            print(json.dumps(cfg.to_dict(), indent=2))
        except ValueError as e:
            logger.error(str(e))
            return 1
    else:
        logger.error(f"Acción desconocida: {args.action}")
        return 1

    return 0


def cmd_info(args: argparse.Namespace) -> int:
    """Muestra información del experimento y del modelo."""
    print("=" * 60)
    print("EXPERIMENTO AI 800M — INFORMACIÓN")
    print("=" * 60)
    print(f"Parámetros totales: {count_total_params():,}")
    print(f"Tamaño estimado en disco: {estimate_disk_size_mb():.2f} MB")
    print()

    spec = build_tensor_spec()
    spec_validation = verify_tensor_spec_consistency(spec)
    print(f"Tensor spec consistente: {spec_validation['consistent']}")
    print(f"Conteo de tensores: {spec_validation['tensor_count']}")
    print()

    model_dir = Path(args.model_dir) if args.model_dir else Path("model")
    weights_file = model_dir / (args.weights_file or "model.safetensors")
    config_file = model_dir / "config.json"

    print(f"Directorio del modelo: {model_dir.resolve()}")
    print(f"Pesos: {weights_file.name} (existe: {weights_file.exists()})")
    print(f"Config: {config_file.name} (existe: {config_file.exists()})")

    if config_file.exists():
        config = ExperimentConfig.load_json(config_file)
        print(f"Vocab size: {config.get('model.vocab_size')}")
        print(f"Hidden size: {config.get('model.hidden_size')}")
        print(f"Layers: {config.get('model.num_hidden_layers')}")
        print(f"Attention heads: {config.get('model.num_attention_heads')}")
        print(f"KV heads: {config.get('model.num_key_value_heads')}")
        print(f"Intermediate size: {config.get('model.intermediate_size')}")

    print("=" * 60)
    return 0


# ---------------------------------------------------------------------------
# Parser de argumentos
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Construye el parser de argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Herramientas CLI para el experimento AI 800M"
    )
    subparsers = parser.add_subparsers(dest="command", help="Comando a ejecutar")

    # validate
    p = subparsers.add_parser("validate", help="Validar modelo y configuración")
    p.add_argument("--model-dir", default="model", help="Directorio del modelo")
    p.add_argument("--weights-file", default="model.safetensors", help="Archivo de pesos")
    p.add_argument("--dynamic", action="store_true", help="Incluir validaciones dinámicas")
    p.add_argument("--output", help="Guardar reporte en JSON")
    p.set_defaults(func=cmd_validate)

    # benchmark
    p = subparsers.add_parser("benchmark", help="Benchmarks de rendimiento")
    p.add_argument("--model-dir", default="model", help="Directorio del modelo")
    p.add_argument("--parts-dir", help="Directorio con las 10 partes del safetensor")
    p.add_argument("--output", help="Guardar reporte en JSON")
    p.set_defaults(func=cmd_benchmark)

    # evaluate
    p = subparsers.add_parser("evaluate", help="Ejecutar suites de evaluación")
    p.add_argument("--model-dir", default="model", help="Directorio del modelo")
    p.add_argument("--weights-file", default="model.safetensors", help="Archivo de pesos")
    p.add_argument(
        "--suites", nargs="+",
        choices=["python", "analysis", "conversation", "induction", "no_collapse"],
        help="Suites a ejecutar",
    )
    p.add_argument("--output", help="Guardar reporte en JSON")
    p.add_argument("--markdown", help="Guardar reporte en Markdown")
    p.set_defaults(func=cmd_evaluate)

    # generate
    p = subparsers.add_parser("generate", help="Generar texto con el modelo")
    p.add_argument("--model-dir", default="model", help="Directorio del modelo")
    p.add_argument("--weights-file", default="model.safetensors", help="Archivo de pesos")
    p.add_argument("--prompt", nargs="+", help="Prompt(s) de entrada")
    p.add_argument("--max-new-tokens", type=int, default=20, help="Tokens máximos a generar")
    p.add_argument("--temperature", type=float, default=0.0, help="Temperatura de muestreo")
    p.add_argument("--repetition-penalty", type=float, default=1.0, help="Penalización de repetición")
    p.add_argument("--output", help="Guardar resultados en JSON")
    p.set_defaults(func=cmd_generate)

    # reassemble
    p = subparsers.add_parser("reassemble", help="Reensamblar las 10 partes del safetensor")
    p.add_argument("--parts-dir", required=True, help="Directorio con las partes")
    p.add_argument("--output", help="Archivo de salida (default: model.safetensors en parts-dir)")
    p.set_defaults(func=cmd_reassemble)

    # synthesize
    p = subparsers.add_parser("synthesize", help="Sintetizar pesos (etapas 0-5)")
    p.add_argument("--output-dir", default="/home/user/ai_experiment", help="Directorio de salida")
    p.set_defaults(func=cmd_synthesize)

    # config
    p = subparsers.add_parser("config", help="Gestionar configuración")
    p.add_argument("action", choices=["show", "validate", "size"], help="Acción")
    p.add_argument("--config-file", help="Archivo de configuración (para validate)")
    p.add_argument("--size-name", help="Nombre de tamaño (para size: 800M, 100M, 1B)")
    p.set_defaults(func=cmd_config)

    # info
    p = subparsers.add_parser("info", help="Información del experimento")
    p.add_argument("--model-dir", default="model", help="Directorio del modelo")
    p.add_argument("--weights-file", default="model.safetensors", help="Archivo de pesos")
    p.set_defaults(func=cmd_info)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Punto de entrada principal de la CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
