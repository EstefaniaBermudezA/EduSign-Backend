"""Banco de pruebas que evalúa las respuestas del servicio LLM contra un gold standard.

Lanza cada pregunta de ``gold_standard.json`` varias veces contra el endpoint
``/ask``, calcula métricas automáticas de formato (idioma, número de oraciones,
saludos, preguntas de vuelta) y latencia, y vuelca los resultados a CSV/JSON en
``results/``. Las columnas de evaluación humana (pertinencia, veracidad, etc.) se
dejan vacías para completarse a mano. Se ejecuta con:
``python run_eval.py [--endpoint URL] [--gold RUTA] [--out DIR]``.
"""

import os
import re
import json
import csv
import time
import argparse
import statistics
from datetime import datetime
from collections import defaultdict

import requests


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
GOLD_PATH = os.path.join(THIS_DIR, "gold_standard.json")
RESULTS_DIR = os.path.join(THIS_DIR, "results")


def _mean(xs):
    """Media aritmética de ``xs``, o 0.0 si la lista está vacía."""
    return statistics.fmean(xs) if xs else 0.0


def _percentile(xs, q):
    """Percentil ``q`` (0-100) de ``xs`` por interpolación lineal; 0.0 si está vacía."""
    if not xs:
        return 0.0
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (q / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] + (s[hi] - s[lo]) * frac


SPANISH_STOPWORDS = {
    "el", "la", "los", "las", "un", "una", "de", "del", "al", "en", "y",
    "que", "es", "son", "fue", "era", "ser", "ha", "para", "por", "con",
    "se", "su", "sus", "le", "les", "no", "yo", "ella", "este", "esta",
    "estos", "estas", "como", "mas", "pero", "tambien", "muy", "ya", "hay",
}

GREETING_PREFIXES = (
    "hola", "buenas", "buen dia", "buen día", "saludos",
    "que tal", "qué tal", "estimado", "estimada",
)


def normalize_text(s):
    """Pasa a minúsculas y elimina tildes y la ñ para comparaciones robustas."""
    s = s.lower()
    for a, b in (("á", "a"), ("é", "e"), ("í", "i"), ("ó", "o"), ("ú", "u"), ("ñ", "n")):
        s = s.replace(a, b)
    return s


def count_sentences(text):
    """Cuenta oraciones no vacías separando por signos de puntuación finales."""
    parts = re.split(r"[.!?]+", text.strip())
    return len([p for p in parts if p.strip()])


def starts_with_greeting(text):
    """Indica si el texto empieza con un saludo (prohibido por el system prompt)."""
    t = normalize_text(text).lstrip()
    return any(t.startswith(g) for g in GREETING_PREFIXES)


def ends_with_question(text):
    """Indica si el texto termina en una pregunta (no debe devolver preguntas)."""
    t = text.strip()
    return bool(t) and (t.endswith("?") or t.endswith("¿"))


def looks_spanish(text):
    """Heurística de idioma: True si aparecen al menos 3 stopwords en español."""
    tokens = re.findall(r"[a-záéíóúñü]+", text.lower())
    if not tokens:
        return False
    count = sum(1 for tok in tokens if normalize_text(tok) in SPANISH_STOPWORDS)
    return count >= 3


def evaluate_format(text):
    """Calcula las métricas automáticas de formato de una respuesta del LLM.

    Returns:
        dict: Conteos (caracteres, palabras, oraciones) y banderas booleanas;
            ``format_ok`` resume si cumple todas las reglas de estilo (español,
            1-3 oraciones, sin saludo y sin pregunta final).
    """
    n_sent = count_sentences(text)
    greet = starts_with_greeting(text)
    q_end = ends_with_question(text)
    es = looks_spanish(text)
    n_words = len(re.findall(r"\w+", text))
    n_chars = len(text)
    format_ok = es and (1 <= n_sent <= 3) and (not greet) and (not q_end)
    return {
        "num_chars": n_chars,
        "num_words": n_words,
        "num_sentences": n_sent,
        "starts_with_greeting": greet,
        "ends_with_question": q_end,
        "looks_spanish": es,
        "format_ok": format_ok,
    }


def call_llm(endpoint, prompt, character, model, temperature, max_tokens, timeout_s):
    """Hace una llamada POST al endpoint ``/ask`` y normaliza el resultado.

    Nunca lanza excepciones de red: ante un error las captura y las refleja en el
    dict devuelto.

    Returns:
        dict: Con ``ok``, ``answer``, latencia reportada por el servidor
            (``latency_server_ms``), latencia total medida en el cliente
            (``latency_total_ms``), ``status_code`` y ``error``.
    """
    payload = {
        "prompt": prompt,
        "character": character,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    t0 = time.perf_counter()
    try:
        r = requests.post(endpoint, json=payload, timeout=timeout_s)
    except requests.RequestException as e:
        return {
            "ok": False, "answer": "",
            "latency_server_ms": -1,
            "latency_total_ms": (time.perf_counter() - t0) * 1000.0,
            "status_code": -1,
            "error": f"RequestException: {e}",
        }
    elapsed_total_ms = (time.perf_counter() - t0) * 1000.0
    if r.status_code != 200:
        return {
            "ok": False, "answer": "",
            "latency_server_ms": -1,
            "latency_total_ms": elapsed_total_ms,
            "status_code": r.status_code,
            "error": r.text[:500],
        }
    data = r.json()
    return {
        "ok": True,
        "answer": data.get("answer", ""),
        "latency_server_ms": data.get("latency_ms", -1),
        "latency_total_ms": elapsed_total_ms,
        "status_code": 200,
        "error": "",
    }


def main():
    """Ejecuta la evaluación completa: lee el gold, llama al LLM y escribe resultados."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--gold", default=GOLD_PATH)
    parser.add_argument("--out", default=RESULTS_DIR)
    parser.add_argument("--timeout", type=float, default=70.0)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    with open(args.gold, encoding="utf-8") as f:
        gold = json.load(f)

    cfg = gold["config"]
    endpoint = args.endpoint or cfg["endpoint_default"]
    model = cfg["model_default"]
    temperature = cfg["temperature_default"]
    max_tokens = cfg["max_tokens_default"]
    n_runs = cfg["n_runs_per_question"]
    send_character_field = cfg.get("send_character_field", True)
    character_prompts = gold.get("character_prompts", {})

    print(f"Endpoint        : {endpoint}")
    print(f"Modelo          : {model}")
    print(f"Temperatura     : {temperature}")
    print(f"Max tokens      : {max_tokens}")
    print(f"Runs por pregunta: {n_runs}")
    print(f"Preguntas       : {len(gold['questions'])}")
    print(f"Salida          : {args.out}")
    print(f"send_character  : {send_character_field}")
    print(f"Personajes (templates): {sorted(character_prompts.keys()) if character_prompts else 'NINGUNO (modo v1)'}")
    print()

    runs_rows = []
    per_q_aggregate = defaultdict(list)
    latency_by_q = defaultdict(list)

    t_global = time.time()
    for q in gold["questions"]:
        qid = q["id"]
        character = q["character"]
        question = q["question"]
        cp = character_prompts.get(character, {}) if character_prompts else {}
        prefix = cp.get("prefix", "")
        suffix = cp.get("suffix", "")
        full_prompt = f"{prefix}{question}{suffix}" if (prefix or suffix) else question
        # Modo v1: la persona viaja embebida en el prompt en lugar del campo character.
        api_character = character if send_character_field else None
        print(f"[{qid:12s}] {character:10s} | {question}")
        for run in range(1, n_runs + 1):
            res = call_llm(endpoint, full_prompt, api_character,
                           model, temperature, max_tokens, args.timeout)
            if not res["ok"]:
                print(f"   run {run}: FAIL ({res['status_code']}) -> {res['error'][:120]}")
                runs_rows.append({
                    "id": qid, "run": run, "character": character,
                    "question": question, "answer": "",
                    "ok": False, "status_code": res["status_code"],
                    "latency_server_ms": res["latency_server_ms"],
                    "latency_total_ms": round(res["latency_total_ms"], 1),
                    "num_chars": 0, "num_words": 0, "num_sentences": 0,
                    "starts_with_greeting": "", "ends_with_question": "",
                    "looks_spanish": "", "format_ok": "",
                    "pertinencia": "", "veracidad_0_2": "",
                    "vocabulario_apropiado": "", "alucinacion": "", "notas": "",
                })
                continue
            fmt = evaluate_format(res["answer"])
            print(f"   run {run}: lat_srv={res['latency_server_ms']:5d} ms  "
                  f"words={fmt['num_words']:3d}  sentences={fmt['num_sentences']}  "
                  f"format_ok={fmt['format_ok']}")
            runs_rows.append({
                "id": qid, "run": run, "character": character,
                "question": question, "answer": res["answer"].replace("\n", " ").strip(),
                "ok": True, "status_code": 200,
                "latency_server_ms": res["latency_server_ms"],
                "latency_total_ms": round(res["latency_total_ms"], 1),
                **fmt,
                "pertinencia": "", "veracidad_0_2": "",
                "vocabulario_apropiado": "", "alucinacion": "", "notas": "",
            })
            per_q_aggregate[qid].append(fmt)
            latency_by_q[qid].append(res["latency_server_ms"])
    elapsed = time.time() - t_global

    # CSV detallado
    runs_csv = os.path.join(args.out, "llm_eval_runs.csv")
    fieldnames = [
        "id", "run", "character", "question", "answer",
        "ok", "status_code",
        "latency_server_ms", "latency_total_ms",
        "num_chars", "num_words", "num_sentences",
        "starts_with_greeting", "ends_with_question",
        "looks_spanish", "format_ok",
        "pertinencia", "veracidad_0_2", "vocabulario_apropiado",
        "alucinacion", "notas",
    ]
    with open(runs_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        w.writeheader()
        for row in runs_rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})

    # CSV resumen por pregunta
    per_q_csv = os.path.join(args.out, "llm_eval_per_question.csv")
    with open(per_q_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow([
            "id", "character", "n_runs",
            "latency_mean_ms", "latency_p95_ms",
            "words_mean", "sentences_mean",
            "pct_format_ok", "pct_greeting", "pct_question_end", "pct_spanish",
        ])
        for q in gold["questions"]:
            qid = q["id"]
            aggs = per_q_aggregate.get(qid, [])
            lats = latency_by_q.get(qid, [])
            if not aggs:
                w.writerow([qid, q["character"], 0, "", "", "", "", "", "", "", ""])
                continue
            n = len(aggs)
            w.writerow([
                qid, q["character"], n,
                round(_mean(lats), 1),
                round(_percentile(lats, 95), 1),
                round(_mean([a["num_words"] for a in aggs]), 1),
                round(_mean([a["num_sentences"] for a in aggs]), 1),
                round(_mean([1 if a["format_ok"] else 0 for a in aggs]) * 100, 1),
                round(_mean([1 if a["starts_with_greeting"] else 0 for a in aggs]) * 100, 1),
                round(_mean([1 if a["ends_with_question"] else 0 for a in aggs]) * 100, 1),
                round(_mean([1 if a["looks_spanish"] else 0 for a in aggs]) * 100, 1),
            ])

    # Metadatos
    meta = {
        "timestamp": datetime.now().isoformat(),
        "endpoint": endpoint,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "n_runs_per_question": n_runs,
        "total_questions": len(gold["questions"]),
        "total_calls": sum(len(per_q_aggregate.get(q["id"], [])) for q in gold["questions"]),
        "elapsed_seconds": round(elapsed, 1),
    }
    meta_path = os.path.join(args.out, "llm_eval_run_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # Resumen final
    all_lats = [v for lats in latency_by_q.values() for v in lats]
    all_format_ok = sum(1 for r in runs_rows if r.get("format_ok") is True)
    all_runs_ok = sum(1 for r in runs_rows if r.get("ok") is True)
    print()
    print()
    print("=" * 60)
    print("RESUMEN")
    print("=" * 60)
    print(f"Llamadas exitosas      : {all_runs_ok}/{len(runs_rows)}")
    if all_lats:
        mean_lat = _mean(all_lats)
        p50_lat = _percentile(all_lats, 50)
        p95_lat = _percentile(all_lats, 95)
        max_lat = max(all_lats)
        print(f"Latencia server (ms)   : mean={mean_lat:.0f}  p50={p50_lat:.0f}  p95={p95_lat:.0f}  max={max_lat:.0f}")
    pct_fmt = 100.0 * all_format_ok / max(all_runs_ok, 1)
    print(f"Format OK              : {all_format_ok}/{all_runs_ok} ({pct_fmt:.1f}%)")
    print(f"Tiempo total           : {elapsed:.0f}s")
    print()
    print(f"Salidas en {args.out}")
    print("Siguiente paso: abrir llm_eval_runs.csv y completar las columnas manuales")


if __name__ == "__main__":
    main()
