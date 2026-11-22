#!/usr/bin/env python3
"""
Programmatic Zero-Training BPE Tokenizer Builder (32,000 tokens: English + Python + Analysis)
Constructs a valid ByteLevel BPE tokenizer without training on any text corpus.
Outputs:
  - tokenizer.json
  - tokenizer_config.json
  - token_metadata.json (explicit structural categories used by the analytical weight synthesizer)
"""
import json
import keyword
import builtins
import math
import statistics
import itertools
import functools
import collections
import re
import os
import sys
import typing
import ast
from pathlib import Path
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors

OUT_DIR = Path("/home/user/ai_experiment")
VOCAB_SIZE = 32000

def bytes_to_unicode():
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))

B2U = bytes_to_unicode()

def to_bytelevel(s: str) -> str:
    return "".join(B2U[b] for b in s.encode("utf-8"))

SPECIAL_TOKENS = [
    "<|bos|>",
    "<|eos|>",
    "<|pad|>",
    "<|unk|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|python|>",
    "<|analysis|>",
    "<|end|>",
]

PYTHON_KEYWORDS = list(keyword.kwlist) + ["match", "case", "type", "self", "cls"]
PYTHON_BUILTINS = [x for x in dir(builtins) if not x.startswith("__")] + [
    "__init__", "__str__", "__repr__", "__len__", "__getitem__", "__setitem__",
    "__iter__", "__next__", "__call__", "__enter__", "__exit__", "__name__", "__main__"
]

PYTHON_STDLIB_AND_DS = []
for mod in (math, statistics, itertools, functools, collections, re, os, sys, typing, ast):
    PYTHON_STDLIB_AND_DS.extend([x for x in dir(mod) if not x.startswith("_")])

PYTHON_EXTRA = [
    "np", "numpy", "pd", "pandas", "torch", "nn", "plt", "scipy", "sklearn",
    "json", "loads", "dumps", "load", "dump", "read", "write", "readlines",
    "append", "extend", "insert", "pop", "remove", "clear", "copy", "keys",
    "values", "items", "update", "get", "setdefault", "split", "join", "strip",
    "lstrip", "rstrip", "replace", "find", "index", "count", "startswith",
    "endswith", "lower", "upper", "title", "format", "encode", "decode",
    "shape", "dtype", "zeros", "ones", "empty", "full", "arange", "linspace",
    "reshape", "transpose", "matmul", "dot", "mean", "std", "var", "median",
    "quantile", "percentile", "sum", "prod", "cumsum", "argmax", "argmin",
    "argsort", "sort", "where", "concatenate", "stack", "vstack", "hstack",
    "dataframe", "DataFrame", "Series", "groupby", "merge", "dropna", "fillna",
    "apply", "describe", "corr", "value_counts", "head", "tail", "iloc", "loc",
    "array", "ndarray", "matrix", "tensor", "vector", "arr", "df", "res", "result",
    "data", "row", "col", "idx", "key", "val", "item", "node", "edge", "graph",
    "tree", "root", "left", "right", "mid", "low", "high", "pivot", "stack",
    "queue", "heap", "visited", "seen", "cache", "memo", "dp", "ans", "out",
    "total", "cnt", "num", "den", "eps", "tol", "step", "steps", "epoch",
    "func", "fn", "args", "kwargs", "err", "msg", "path", "url", "cfg", "config",
    "assert_equal", "unittest", "pytest", "dataclass", "Enum", "Path", "Counter",
    "defaultdict", "deque", "lru_cache", "partial", "reduce", "accumulate",
    "combinations", "permutations", "product", "chain", "zip_longest",
]

PYTHON_SYNTAX_BLOCKS = [
    "    ", "        ", "            ", "                ",
    "\n    ", "\n        ", "\n            ", "\n\n",
    "():", "):", "]:", "}:", " -> ", " == ", " != ", " <= ", " >= ",
    " += ", " -= ", " *= ", " /= ", " //= ", " %= ", " **= ", " := ",
    " = ", " + ", " - ", " * ", " / ", " // ", " % ", " ** ",
    "()", "[]", "{}", "''", '""', '"""', "'''", "f\"", "f'", "r\"",
    " == None", " is None", " is not None", " in range(", "for i in range(",
    "if __name__ == '__main__':", 'if __name__ == "__main__":',
    "def ", "class ", "return ", "import ", "from ", "elif ", "else:",
    "try:", "except ", "finally:", "with ", "while ", "for ", "lambda ",
    "raise ", "yield ", "async def ", "await ", "pass\n", "break\n", "continue\n",
]

ANALYSIS_WORDS = [
    "analysis", "analyze", "analytical", "evaluate", "evaluation", "assess",
    "assessment", "compare", "comparison", "contrast", "correlate", "correlation",
    "causation", "cause", "effect", "hypothesis", "evidence", "conclusion",
    "conclude", "infer", "inference", "deduce", "deduction", "induce", "induction",
    "therefore", "thus", "hence", "because", "since", "however", "nevertheless",
    "moreover", "furthermore", "additionally", "consequently", "specifically",
    "notably", "importantly", "crucially", "essentially", "fundamentally",
    "first", "second", "third", "fourth", "finally", "initially", "subsequently",
    "step", "steps", "phase", "stage", "factor", "factors", "variable", "variables",
    "parameter", "parameters", "metric", "metrics", "measure", "measurement",
    "distribution", "variance", "deviation", "standard", "average", "mean",
    "median", "mode", "range", "quartile", "percentile", "outlier", "outliers",
    "anomaly", "anomalies", "trend", "trends", "pattern", "patterns", "signal",
    "noise", "ratio", "proportion", "percentage", "rate", "growth", "decline",
    "increase", "decrease", "minimum", "maximum", "optimal", "optimize",
    "optimization", "complexity", "time", "space", "memory", "latency",
    "throughput", "scalability", "bottleneck", "tradeoff", "tradeoffs",
    "constraint", "constraints", "assumption", "assumptions", "limitation",
    "limitations", "validity", "valid", "invalid", "consistent", "inconsistent",
    "contradiction", "contradicts", "implies", "equivalent", "necessary",
    "sufficient", "condition", "conditions", "probability", "likelihood",
    "confidence", "significance", "significant", "bias", "error", "errors",
    "residual", "accuracy", "precision", "recall", "benchmark", "baseline",
    "ablation", "experiment", "experimental", "observation", "observe",
    "empirical", "theoretical", "mathematical", "matrix", "eigenvalue",
    "eigenvector", "orthogonal", "projection", "subspace", "rank", "norm",
    "entropy", "gradient", "convergence", "divergence", "stability", "unstable",
    "linear", "nonlinear", "logarithmic", "exponential", "polynomial", "quadratic",
    "constant", "asymptotic", "algorithm", "heuristic", "deterministic",
    "stochastic", "decomposition", "breakdown", "summary", "summarize", "insight",
    "insights", "finding", "findings", "risk", "impact", "priority", "root",
    "diagnostic", "diagnose", "debug", "debugging", "traceback", "verify",
    "verification", "validate", "validation", "proof", "prove", "theorem",
    "lemma", "axiom", "logic", "logical", "reasoning", "reason", "structure",
    "structural", "system", "systematic", "framework", "architecture", "model",
]

CONVERSATION_AND_CORE_ENGLISH = [
    # Greetings & minimal conversational fluency
    "hello", "hi", "hey", "greetings", "welcome", "goodbye", "bye", "thanks",
    "thank", "please", "sorry", "yes", "no", "maybe", "sure", "okay", "ok",
    "good", "great", "fine", "well", "help", "assist", "assistant", "user",
    "question", "answer", "explain", "describe", "clarify", "understand",
    "understanding", "meaning", "know", "think", "believe", "seem", "appear",
    "can", "could", "would", "should", "will", "shall", "may", "might", "must",
    "do", "does", "did", "done", "doing", "make", "made", "making", "create",
    "created", "creating", "build", "built", "building", "write", "wrote",
    "written", "writing", "read", "reading", "run", "running", "test", "testing",
    "check", "checking", "show", "showing", "give", "given", "take", "taken",
    "use", "used", "using", "work", "works", "working", "need", "needs", "want",
    "have", "has", "had", "having", "be", "am", "is", "are", "was", "were",
    "been", "being", "become", "becomes", "get", "got", "getting", "find",
    "found", "keep", "kept", "let", "put", "set", "start", "begin", "end",
    "stop", "continue", "change", "changed", "edit", "edited", "modify",
    # Pronouns, prepositions, conjunctions, articles, WH-words
    "I", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their", "mine", "yours", "ours", "theirs",
    "this", "that", "these", "those", "here", "there", "where", "when", "why",
    "how", "what", "which", "who", "whom", "whose", "whether", "while", "if",
    "then", "else", "for", "from", "to", "in", "on", "at", "by", "with", "of",
    "as", "into", "onto", "upon", "about", "above", "below", "under", "over",
    "between", "among", "through", "during", "before", "after", "without",
    "within", "against", "across", "around", "near", "far", "both", "each",
    "every", "all", "any", "some", "none", "no", "not", "only", "just", "also",
    "very", "more", "most", "less", "least", "much", "many", "few", "several",
    "other", "another", "such", "same", "different", "similar", "new", "old",
    "large", "small", "big", "little", "long", "short", "high", "low", "fast",
    "slow", "early", "late", "next", "previous", "current", "main", "key",
    "major", "minor", "clear", "simple", "complex", "easy", "hard", "difficult",
    "possible", "impossible", "likely", "unlikely", "true", "false", "correct",
    "incorrect", "right", "wrong", "better", "best", "worse", "worst", "high",
    "higher", "highest", "lower", "lowest", "greater", "smaller", "equal",
    "a", "an", "the", "and", "or", "but", "nor", "so", "yet",
    # General technical & English nouns/verbs
    "code", "program", "programming", "script", "function", "method", "class",
    "object", "instance", "attribute", "property", "module", "package", "library",
    "file", "folder", "directory", "line", "lines", "block", "statement",
    "expression", "value", "values", "type", "types", "string", "integer",
    "float", "boolean", "list", "tuple", "dictionary", "set", "array", "table",
    "column", "columns", "row", "rows", "record", "records", "field", "fields",
    "index", "key", "pair", "element", "elements", "item", "items", "loop",
    "iteration", "iterate", "recursion", "recursive", "branch", "condition",
    "input", "inputs", "output", "outputs", "argument", "arguments", "return",
    "returns", "print", "prints", "exception", "warning", "bug", "bugs", "fix",
    "fixed", "issue", "issues", "problem", "problems", "solution", "solutions",
    "approach", "strategy", "design", "pattern", "state", "process", "task",
    "tasks", "goal", "objective", "case", "cases", "example", "examples",
    "sample", "samples", "detail", "details", "part", "parts", "piece", "point",
    "points", "way", "ways", "form", "level", "levels", "order", "sequence",
    "series", "group", "groups", "cluster", "clusters", "pair", "pairs",
    "network", "layer", "layers", "weight", "weights", "bias", "attention",
    "head", "heads", "token", "tokens", "embedding", "embeddings", "vector",
    "vectors", "dimension", "dimensions", "feature", "features", "score",
    "scores", "loss", "cost", "reward", "value", "signal", "state", "states",
    "context", "prompt", "response", "text", "word", "words", "sentence",
    "sentences", "language", "syntax", "semantics", "grammar", "rule", "rules",
    "logic", "math", "equation", "formula", "number", "numbers", "count",
    "counts", "sum", "product", "difference", "quotient", "remainder", "power",
    "root", "log", "exponent", "scale", "offset", "threshold", "limit",
    "bound", "bounds", "interval", "window", "span", "size", "length", "width",
    "height", "depth", "capacity", "speed", "performance", "efficiency",
    "quality", "reliability", "robustness", "precision", "uncertainty",
    "today", "now", "always", "never", "often", "sometimes", "usually",
    "generally", "typically", "specifically", "exactly", "approximately",
    "roughly", "nearly", "almost", "completely", "partially", "entirely",
    "directly", "indirectly", "automatically", "manually", "analytically",
    "able", "ready", "sure", "certain", "clear", "obvious", "evident",
    "important", "useful", "helpful", "effective", "efficient", "accurate",
    "reliable", "consistent", "relevant", "critical", "essential", "primary",
    "secondary", "final", "initial", "intermediate", "internal", "external",
    "local", "global", "public", "private", "static", "dynamic", "abstract",
    "concrete", "formal", "informal", "practical", "theoretical", "basic",
    "advanced", "standard", "custom", "default", "optional", "required",
    "available", "missing", "empty", "full", "zero", "one", "two", "three",
    "four", "five", "six", "seven", "eight", "nine", "ten", "hundred", "thousand",
    "million", "billion", "firstly", "secondly", "thirdly", "lastly",
    # Common verbs + inflections generator seeds
    "compute", "calculate", "evaluate", "estimate", "predict", "classify",
    "cluster", "filter", "sort", "search", "match", "parse", "extract",
    "transform", "convert", "encode", "decode", "compress", "expand",
    "merge", "split", "combine", "separate", "compare", "rank", "select",
    "choose", "decide", "determine", "identify", "detect", "discover",
    "explore", "inspect", "examine", "investigate", "study", "review",
    "report", "document", "log", "track", "monitor", "measure", "count",
    "aggregate", "summarize", "reduce", "map", "apply", "execute", "run",
    "call", "invoke", "pass", "return", "yield", "raise", "catch", "handle",
    "prevent", "avoid", "allow", "enable", "disable", "support", "require",
    "depend", "relate", "connect", "link", "bind", "assign", "store", "save",
    "load", "fetch", "send", "receive", "update", "delete", "remove", "add",
    "insert", "append", "replace", "modify", "adjust", "tune", "edit",
    "improve", "optimize", "simplify", "refactor", "clean", "format",
    "validate", "verify", "confirm", "prove", "disprove", "fail", "succeed",
    "break", "crash", "halt", "pause", "resume", "repeat", "loop", "skip",
    "wait", "sleep", "close", "open", "exit", "quit", "start", "finish",
    "solve", "resolve", "answer", "reply", "respond", "ask", "request",
    "suggest", "recommend", "propose", "assume", "suppose", "expect",
    "observe", "notice", "see", "look", "listen", "hear", "speak", "talk",
    "say", "tell", "state", "mention", "note", "highlight", "emphasize",
    "indicate", "imply", "demonstrate", "illustrate", "represent", "define",
    "specify", "include", "contain", "consist", "comprise", "involve",
    "affect", "influence", "cause", "lead", "produce", "generate", "derive",
    "obtain", "achieve", "reach", "maintain", "preserve", "ensure", "guarantee",
]

# Common English/code affixes & subwords
SUBWORDS = [
    "ing", "ed", "er", "ers", "est", "ly", "tion", "ation", "sion", "ment",
    "ness", "ity", "ty", "able", "ible", "ful", "less", "ous", "ive", "al",
    "ial", "ic", "ical", "ize", "ise", "ify", "ate", "en", "es", "s",
    "un", "re", "in", "im", "dis", "non", "pre", "post", "sub", "super",
    "inter", "trans", "over", "under", "out", "up", "down", "auto", "multi",
    "micro", "macro", "meta", "self", "semi", "anti", "pro", "co", "de",
    "th", "he", "an", "re", "er", "on", "at", "en", "nd", "ti", "es", "or",
    "te", "of", "ed", "is", "it", "al", "ar", "st", "to", "nt", "ng", "se",
    "ha", "as", "ou", "io", "le", "ve", "co", "me", "de", "hi", "ri", "ro",
    "ic", "ne", "ea", "ra", "ce", "li", "ch", "ll", "be", "ma", "si", "om",
    "ur", "ca", "el", "ta", "la", "ns", "di", "fo", "ho", "pe", "ec", "pr",
    "no", "ct", "us", "ac", "ot", "il", "tr", "ly", "nc", "et", "ut", "ss",
    "so", "rs", "un", "lo", "wa", "ge", "ie", "wh", "ee", "wi", "em", "ad",
    "ol", "rt", "po", "we", "na", "ul", "ni", "ts", "mo", "ow", "pa", "ur",
    "py", "np", "pd", "df", "fn", "id", "io", "os", "re", "db", "ai", "ml",
    "__", "._", "_id", "_in", "_to", "_by", "_of", "_is", "_at", "_on",
    "get_", "set_", "is_", "has_", "to_", "from_", "max_", "min_", "num_",
    "init", "iter", "next", "prev", "curr", "temp", "tmp", "buf", "ptr",
    "src", "dst", "req", "res", "resp", "ctx", "env", "cfg", "opt", "arg",
    "kw", "idx", "pos", "len", "dim", "vec", "mat", "arr", "str", "int",
    "flt", "obj", "cls", "mod", "pkg", "lib", "api", "cli", "gui", "url",
    "uri", "http", "https", "utf", "ascii", "json", "csv", "yaml", "xml",
    "html", "sql", "regex", "ast", "bpe", "mlp", "gqa", "rope", "rms",
    "silu", "gelu", "softmax", "logits", "norm", "head", "attn", "proj",
    "gate", "up", "down", "embed", "vocab", "batch", "seq", "layer",
]

# Common broader English nouns/adjectives/verbs for vocabulary breadth
EXTRA_ENGLISH_WORDS = [
    "ability", "absence", "access", "account", "action", "activity", "actor",
    "addition", "address", "advantage", "advice", "agency", "agent", "agreement",
    "air", "amount", "animal", "area", "argument", "arm", "art", "article",
    "aspect", "attempt", "audience", "author", "authority", "balance", "bank",
    "base", "basis", "behavior", "benefit", "bill", "bird", "bit", "block",
    "blood", "board", "body", "book", "border", "box", "boy", "brain", "branch",
    "break", "bridge", "brother", "budget", "building", "business", "button",
    "buyer", "camera", "campaign", "cancer", "candidate", "capital", "car",
    "card", "care", "career", "category", "cell", "center", "central", "century",
    "chair", "challenge", "chance", "channel", "chapter", "character", "charge",
    "chart", "check", "child", "children", "choice", "church", "circle", "city",
    "claim", "client", "climate", "clock", "cloud", "club", "coach", "coast",
    "coffee", "collection", "college", "color", "command", "comment", "commerce",
    "commission", "committee", "communication", "community", "company", "concept",
    "concern", "concert", "conference", "conflict", "congress", "connection",
    "consensus", "consequence", "consideration", "consumer", "contact", "content",
    "context", "contract", "control", "conversation", "core", "corner", "cost",
    "council", "country", "county", "couple", "course", "court", "cover",
    "credit", "crime", "crisis", "criterion", "critic", "criticism", "crop",
    "crowd", "culture", "cup", "currency", "customer", "cycle", "damage",
    "danger", "dark", "database", "date", "daughter", "day", "deal", "dealer",
    "death", "debate", "debt", "decade", "decision", "defense", "degree",
    "delay", "delivery", "demand", "democracy", "department", "deposit",
    "depression", "description", "desert", "desk", "device", "dialogue",
    "diamond", "diet", "direction", "director", "discipline", "discount",
    "discussion", "disease", "disk", "display", "distance", "district", "doctor",
    "document", "dog", "dollar", "domain", "door", "doubt", "draft", "drama",
    "dream", "dress", "drink", "drive", "driver", "drop", "drug", "duty",
    "earth", "ease", "east", "economy", "edge", "edition", "editor", "education",
    "effort", "egg", "election", "electricity", "emotion", "emphasis", "employee",
    "employer", "employment", "energy", "engine", "engineer", "engineering",
    "enterprise", "entertainment", "enthusiasm", "entry", "environment", "episode",
    "equipment", "era", "escape", "essay", "establishment", "estate", "estimate",
    "ethics", "event", "evening", "evidence", "evolution", "exam", "examination",
    "example", "exchange", "excitement", "excuse", "executive", "exercise",
    "exhibition", "existence", "expansion", "expectation", "expense", "experience",
    "expert", "explanation", "exposure", "extension", "extent", "eye", "face",
    "facility", "fact", "factory", "failure", "faith", "fall", "fame", "family",
    "fan", "farm", "farmer", "fashion", "father", "fault", "fear", "feature",
    "feedback", "feeling", "female", "fence", "festival", "fiction", "figure",
    "film", "finance", "finger", "fire", "firm", "fish", "flight", "floor",
    "flow", "flower", "focus", "food", "foot", "football", "force", "forest",
    "fortune", "foundation", "frame", "framework", "freedom", "friend", "front",
    "fruit", "fuel", "fun", "fund", "funding", "future", "gain", "gallery",
    "game", "gap", "garage", "garden", "gas", "gate", "gear", "gene", "general",
    "generation", "genius", "genre", "gesture", "gift", "girl", "glass", "globe",
    "gold", "golf", "government", "governor", "grace", "grade", "grain", "grand",
    "grant", "grass", "gravity", "green", "ground", "guarantee", "guard", "guess",
    "guest", "guide", "guideline", "guitar", "guy", "habit", "hair", "half",
    "hall", "hand", "handle", "harm", "hat", "head", "health", "hearing", "heart",
    "heat", "heaven", "height", "hell", "hero", "highway", "hill", "history",
    "hit", "hold", "hole", "holiday", "home", "homework", "honey", "honor",
    "hook", "hope", "horizon", "horse", "hospital", "host", "hotel", "hour",
    "house", "housing", "human", "humor", "hunger", "hunt", "husband", "ice",
    "idea", "ideal", "identity", "image", "imagination", "impact", "importance",
    "impression", "improvement", "incident", "income", "independence", "index",
    "indication", "individual", "industry", "inflation", "influence", "information",
    "initiative", "injury", "innovation", "inquiry", "insect", "inside", "insight",
    "inspection", "inspector", "installation", "instance", "institution",
    "instruction", "instrument", "insurance", "intelligence", "intention",
    "interaction", "interest", "interior", "interpretation", "interview",
    "introduction", "investment", "investor", "invitation", "iron", "island",
    "item", "jacket", "job", "joint", "joke", "journal", "journey", "joy",
    "judge", "judgment", "juice", "jump", "jury", "justice", "kid", "king",
    "kitchen", "knee", "knife", "knowledge", "lab", "label", "labor", "lack",
    "ladder", "lady", "lake", "land", "landscape", "language", "lap", "laugh",
    "law", "lawyer", "layer", "lead", "leader", "leadership", "league", "learning",
    "leather", "leave", "lecture", "leg", "legend", "legislation", "lesson",
    "letter", "level", "liberty", "library", "license", "life", "lift", "light",
    "limit", "link", "lip", "liquid", "list", "literature", "load", "loan",
    "location", "lock", "log", "logic", "look", "loss", "love", "luck", "lunch",
    "machine", "magazine", "magic", "mail", "maintenance", "majority", "maker",
    "male", "mall", "man", "management", "manager", "manner", "manufacturer",
    "manufacturing", "map", "margin", "mark", "market", "marketing", "marriage",
    "mask", "mass", "master", "match", "material", "math", "matter", "meal",
    "meaning", "measure", "measurement", "meat", "mechanism", "media", "medicine",
    "medium", "meeting", "member", "membership", "memory", "mention", "menu",
    "merchant", "mercy", "message", "metal", "method", "middle", "midnight",
    "might", "mile", "milk", "mind", "mine", "mineral", "minimum", "minister",
    "ministry", "minor", "minority", "minute", "mirror", "mission", "mistake",
    "mix", "mixture", "mode", "model", "moment", "money", "monitor", "month",
    "mood", "moon", "moral", "morning", "mortgage", "mother", "motion", "motor",
    "mountain", "mouse", "mouth", "move", "movement", "movie", "mud", "muscle",
    "museum", "music", "mystery", "myth", "nail", "name", "narrative", "nation",
    "nature", "navigation", "neck", "need", "negative", "neighbor", "neighborhood",
    "nerve", "net", "network", "news", "newspaper", "night", "noise", "nomination",
    "north", "nose", "note", "nothing", "notice", "notion", "novel", "number",
    "nurse", "nut", "object", "objective", "obligation", "observation",
    "observer", "occasion", "ocean", "offer", "office", "officer", "official",
    "oil", "operation", "opinion", "opponent", "opportunity", "opposition",
    "option", "orange", "order", "organization", "origin", "outcome", "output",
    "outside", "oven", "owner", "ownership", "pace", "pack", "package", "page",
    "pain", "paint", "painter", "painting", "pair", "palace", "palm", "pan",
    "panel", "panic", "paper", "parent", "park", "parking", "part", "participant",
    "particle", "particular", "partner", "partnership", "party", "pass", "passage",
    "passenger", "passion", "past", "path", "patience", "patient", "pattern",
    "pause", "pay", "payment", "peace", "peak", "pen", "penalty", "pencil",
    "people", "pepper", "perception", "performance", "period", "permission",
    "person", "personality", "perspective", "phase", "phenomenon", "philosophy",
    "phone", "photo", "photograph", "phrase", "physics", "piano", "pick",
    "picture", "pie", "piece", "pig", "pile", "pilot", "pin", "pipe", "pitch",
    "place", "plan", "plane", "planet", "planning", "plant", "plastic", "plate",
    "platform", "play", "player", "pleasure", "plenty", "plot", "plus", "pocket",
    "poem", "poet", "poetry", "point", "pole", "police", "policy", "politics",
    "poll", "pollution", "pool", "pop", "population", "port", "portion", "portrait",
    "pose", "position", "positive", "possession", "possibility", "post", "pot",
    "potato", "potential", "pound", "poverty", "powder", "power", "practice",
    "praise", "prayer", "prediction", "preference", "premise", "preparation",
    "presence", "present", "presentation", "president", "press", "pressure",
    "price", "pride", "priest", "primary", "prime", "prince", "princess",
    "principle", "print", "prior", "priority", "prison", "privacy", "prize",
    "problem", "procedure", "process", "produce", "producer", "product",
    "production", "profession", "professional", "professor", "profile", "profit",
    "program", "progress", "project", "promise", "promotion", "proof", "property",
    "proportion", "proposal", "prospect", "protection", "protein", "protest",
    "provider", "province", "provision", "psychology", "public", "publication",
    "publicity", "publisher", "pull", "pulse", "pump", "punch", "pupil",
    "purchase", "purpose", "push", "quality", "quantity", "quarter", "queen",
    "question", "queue", "quiet", "quote", "race", "radio", "rail", "rain",
    "raise", "range", "rank", "rate", "ratio", "raw", "reach", "reaction",
    "reader", "reading", "reality", "reason", "reasoning", "recall", "receipt",
    "receiver", "reception", "recipe", "recognition", "recommendation", "record",
    "recording", "recovery", "reduction", "reference", "reflection", "reform",
    "refugee", "regard", "regime", "region", "register", "regulation", "relation",
    "relationship", "relative", "release", "relevance", "relief", "religion",
    "remark", "remedy", "reminder", "remote", "removal", "rent", "repair",
    "repeat", "replacement", "reply", "report", "reporter", "representation",
    "representative", "republic", "reputation", "request", "requirement",
    "rescue", "research", "researcher", "reserve", "resident", "resistance",
    "resolution", "resolve", "resort", "resource", "respect", "response",
    "responsibility", "rest", "restaurant", "restriction", "result", "retail",
    "retirement", "return", "reveal", "revenue", "review", "revolution", "reward",
    "rhythm", "rice", "rich", "ride", "ring", "rise", "risk", "ritual", "river",
    "road", "rock", "role", "roll", "roof", "room", "root", "rope", "rose",
    "rough", "round", "route", "routine", "row", "royal", "rub", "ruin", "rule",
    "run", "runner", "rush", "sadness", "safe", "safety", "sail", "salad",
    "salary", "sale", "salt", "sample", "sand", "satellite", "satisfaction",
    "save", "saving", "scale", "scan", "scandal", "scene", "schedule", "scheme",
    "scholar", "scholarship", "school", "science", "scientist", "scope", "score",
    "screen", "script", "sea", "search", "season", "seat", "second", "secret",
    "secretary", "section", "sector", "security", "seed", "seek", "segment",
    "selection", "self", "seller", "senate", "senator", "sense", "sentence",
    "sequence", "series", "servant", "service", "session", "setting", "settlement",
    "shade", "shadow", "shake", "shall", "shape", "share", "shareholder", "shark",
    "sheet", "shelf", "shell", "shelter", "shift", "shine", "ship", "shirt",
    "shock", "shoe", "shoot", "shop", "shopping", "shore", "short", "shot",
    "shoulder", "show", "shower", "shut", "sick", "side", "sigh", "sight",
    "sign", "signal", "signature", "significance", "silence", "silver", "simple",
    "sin", "singer", "single", "sink", "sir", "sister", "site", "situation",
    "size", "skill", "skin", "sky", "slave", "sleep", "slice", "slide", "slight",
    "slip", "slope", "slot", "slow", "small", "smart", "smell", "smile", "smoke",
    "smooth", "snap", "snow", "soap", "soccer", "social", "society", "sock",
    "soft", "software", "soil", "solar", "soldier", "solid", "solution", "solve",
    "somebody", "someone", "something", "sometimes", "somewhat", "somewhere",
    "son", "song", "soon", "sort", "soul", "sound", "soup", "source", "south",
    "space", "span", "spare", "spark", "speaker", "special", "specialist",
    "species", "specific", "speech", "speed", "spell", "spend", "sphere",
    "spirit", "split", "spokesman", "sport", "spot", "spread", "spring",
    "square", "squeeze", "stability", "stable", "staff", "stage", "stair",
    "stake", "stamp", "stand", "standard", "star", "start", "state", "statement",
    "station", "statistic", "statistics", "status", "stay", "steady", "steal",
    "steam", "steel", "step", "stick", "still", "stock", "stomach", "stone",
    "stop", "storage", "store", "storm", "story", "strain", "strand", "strange",
    "stranger", "strategy", "stream", "street", "strength", "stress", "stretch",
    "strike", "string", "strip", "stroke", "strong", "structure", "struggle",
    "student", "studio", "study", "stuff", "style", "subject", "submit",
    "substance", "substitute", "success", "such", "sudden", "suffer", "sugar",
    "suggestion", "suit", "sum", "summary", "summer", "sun", "super", "supply",
    "support", "surface", "surgery", "surprise", "survey", "survival", "suspect",
    "suspend", "sustain", "swear", "sweep", "sweet", "swim", "swing", "switch",
    "symbol", "symptom", "system", "table", "tactic", "tail", "take", "tale",
    "talent", "talk", "tall", "tank", "tap", "tape", "target", "task", "taste",
    "tax", "tea", "teach", "teacher", "teaching", "team", "tear", "tech",
    "technique", "technology", "teen", "telephone", "television", "tell",
    "temperature", "temple", "tendency", "tennis", "tension", "tent", "term",
    "terms", "terrain", "test", "text", "texture", "thank", "thanks", "theater",
    "theme", "theory", "therapy", "there", "therefore", "thick", "thin", "thing",
    "think", "thinking", "third", "thought", "thousand", "thread", "threat",
    "threshold", "throat", "through", "throw", "thumb", "ticket", "tide", "tie",
    "tight", "till", "time", "tiny", "tip", "tire", "tissue", "title", "toast",
    "tobacco", "today", "toe", "together", "toilet", "token", "tolerance",
    "tomorrow", "tone", "tongue", "tonight", "tool", "tooth", "top", "topic",
    "total", "touch", "tough", "tour", "tourist", "tournament", "toward", "tower",
    "town", "toy", "trace", "track", "trade", "tradition", "traffic", "tragedy",
    "trail", "train", "trainer", "training", "trait", "transaction", "transfer",
    "transform", "transition", "translate", "transmission", "transport", "trap",
    "trash", "travel", "tray", "treat", "treatment", "treaty", "tree", "trend",
    "trial", "tribe", "trick", "trigger", "trip", "triumph", "troop", "trouble",
    "truck", "true", "trust", "truth", "try", "tube", "tuck", "tuition", "tune",
    "tunnel", "turn", "twin", "twist", "type", "typical", "ugly", "ultimate",
    "umbrella", "uncle", "under", "understanding", "uniform", "union", "unique",
    "unit", "universe", "university", "unknown", "unless", "unlike", "unlikely",
    "until", "unusual", "update", "upgrade", "upper", "urban", "urge", "usage",
    "use", "user", "usual", "utility", "vacation", "vaccine", "vacuum", "valid",
    "valley", "valuable", "value", "van", "variable", "variation", "variety",
    "various", "vast", "vector", "vegetable", "vehicle", "venture", "version",
    "versus", "vessel", "veteran", "victim", "victory", "video", "view",
    "viewer", "village", "violation", "violence", "virtue", "virus", "vision",
    "visit", "visitor", "visual", "vital", "voice", "volume", "volunteer",
    "vote", "voter", "wage", "wait", "wake", "walk", "wall", "wander", "want",
    "war", "warm", "warn", "warning", "wash", "waste", "watch", "water", "wave",
    "way", "weak", "wealth", "weapon", "wear", "weather", "web", "website",
    "wedding", "week", "weekend", "weight", "welcome", "welfare", "well", "west",
    "western", "wet", "whale", "what", "whatever", "wheat", "wheel", "when",
    "whenever", "where", "whereas", "wherever", "whether", "which", "while",
    "whisper", "white", "who", "whoever", "whole", "whom", "whose", "why",
    "wide", "widely", "widow", "width", "wife", "wild", "will", "willing",
    "win", "wind", "window", "wine", "wing", "winner", "winter", "wire",
    "wisdom", "wise", "wish", "with", "within", "without", "witness", "wolf",
    "woman", "wonder", "wood", "wooden", "wool", "word", "work", "worker",
    "working", "workshop", "world", "worry", "worth", "would", "wound", "wrap",
    "write", "writer", "writing", "wrong", "yard", "year", "yellow", "yes",
    "yesterday", "yet", "yield", "you", "young", "your", "yours", "yourself",
    "youth", "zone",
]


def build():
    vocab = {}
    merges = []
    categories = {}  # token_id -> category label

    def add_token(tok: str, cat: str):
        if tok not in vocab and len(vocab) < VOCAB_SIZE:
            tid = len(vocab)
            vocab[tok] = tid
            categories[tid] = cat
            return tid
        return vocab.get(tok)

    # 1. Special tokens
    for st in SPECIAL_TOKENS:
        add_token(st, "special")

    # 2. 256 byte-level base tokens
    for b in range(256):
        ch = B2U[b]
        add_token(ch, "byte")

    merge_ranks = {}

    def apply_existing_merges(syms):
        while len(syms) > 1:
            best_rank = None
            best_idx = -1
            for i in range(len(syms) - 1):
                r = merge_ranks.get((syms[i], syms[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank = r
                    best_idx = i
            if best_idx == -1:
                break
            merged = syms[best_idx] + syms[best_idx + 1]
            syms = syms[:best_idx] + [merged] + syms[best_idx + 2:]
        return syms

    def add_word_with_merges(raw: str, cat: str):
        if len(vocab) >= VOCAB_SIZE:
            return
        bl = to_bytelevel(raw)
        if not bl or bl in vocab:
            if bl in vocab and categories.get(vocab[bl]) == "subword":
                categories[vocab[bl]] = cat
            return
        syms = list(bl)
        for ch in syms:
            if ch not in vocab:
                return
        # First collapse using already-registered merges so priority order stays 100% consistent
        syms = apply_existing_merges(syms)
        while len(syms) > 1:
            if len(vocab) >= VOCAB_SIZE:
                return
            a, b = syms[0], syms[1]
            merged = a + b
            if merged not in vocab:
                if len(vocab) >= VOCAB_SIZE:
                    return
                inter_cat = cat if len(syms) == 2 else "subword"
                add_token(merged, inter_cat)
                merge_ranks[(a, b)] = len(merges)
                merges.append((a, b))
            syms = apply_existing_merges([merged] + syms[2:])
        if syms[0] in vocab:
            categories[vocab[syms[0]]] = cat

    # Helper to add raw + space-prefixed + Title-case forms
    def add_variants(words, cat, include_space=True, include_title=False):
        for w in words:
            add_word_with_merges(w, cat)
            if include_space:
                add_word_with_merges(" " + w, cat)
            if include_title and w and w[0].islower():
                tw = w[0].upper() + w[1:]
                add_word_with_merges(tw, cat)
                if include_space:
                    add_word_with_merges(" " + tw, cat)

    # 3. High-priority Python syntax blocks & keywords
    for blk in PYTHON_SYNTAX_BLOCKS:
        add_word_with_merges(blk, "python_syntax")
    add_variants(PYTHON_KEYWORDS, "python_kw", include_space=True)
    add_variants(PYTHON_BUILTINS, "python_builtin", include_space=True)
    add_variants(PYTHON_STDLIB_AND_DS, "python_lib", include_space=True)
    add_variants(PYTHON_EXTRA, "python_id", include_space=True)

    # 4. Analytical & reasoning tokens
    add_variants(ANALYSIS_WORDS, "analysis", include_space=True, include_title=True)

    # 5. Conversational & core English
    add_variants(CONVERSATION_AND_CORE_ENGLISH, "english_core", include_space=True, include_title=True)

    # 6. Subwords & affixes
    add_variants(SUBWORDS, "subword", include_space=True)

    # 7. Numbers 0..999 (bare and space-prefixed)
    for n in range(1000):
        add_word_with_merges(str(n), "number")
        add_word_with_merges(" " + str(n), "number")

    # 8. Extra English words + morphological suffixes (s, ed, ing, ly, er)
    add_variants(EXTRA_ENGLISH_WORDS, "english_ext", include_space=True, include_title=True)
    morph_base = list(dict.fromkeys(ANALYSIS_WORDS + CONVERSATION_AND_CORE_ENGLISH + EXTRA_ENGLISH_WORDS))
    for w in morph_base:
        if len(vocab) >= VOCAB_SIZE:
            break
        if w.isalpha() and len(w) >= 3:
            for suf in ("s", "es", "ed", "ing", "ly", "er", "ers", "ment", "tion", "ness", "ity", "able"):
                cand = w + suf
                add_word_with_merges(cand, "english_morph")
                add_word_with_merges(" " + cand, "english_morph")
                if len(vocab) >= VOCAB_SIZE:
                    break

    # 9. Python method call patterns (.append, .split, etc.) & snake_case combos
    py_methods = [
        ".append(", ".extend(", ".items()", ".keys()", ".values()", ".split(",
        ".join(", ".strip()", ".replace(", ".format(", ".get(", ".pop(",
        ".sort()", ".copy()", ".mean()", ".std()", ".sum()", ".reshape(",
        ".astype(", ".read()", ".write(", ".close()", ".encode(", ".decode(",
        "self.", "np.", "pd.", "math.", "os.", "sys.", "re.", "json.",
    ]
    for pm in py_methods:
        add_word_with_merges(pm, "python_syntax")
        add_word_with_merges(" " + pm, "python_syntax")

    # 10. Common punctuation & whitespace combinations
    punct_combos = [
        ", ", ". ", ": ", "; ", "? ", "! ", "... ", ":\n", ".\n", ",\n", ")\n", "]\n", "}\n",
        " (", " [", " {", " '", ' "', " #", "\n# ", "\n\n# ", " ->", " =>", "==", "!=", "<=", ">=",
        "//", "**", "/*", "*/", "---", "===", "***", "```python\n", "```\n", "```",
    ]
    for p in punct_combos:
        add_word_with_merges(p, "punct")

    # 11. Deterministically fill remaining slots up to exact 32,000 with useful 2-char / 3-char English & Python bigrams
    alpha = "abcdefghijklmnopqrstuvwxyz_"
    space_g = to_bytelevel(" ")
    if len(vocab) < VOCAB_SIZE:
        for c1 in alpha:
            for c2 in alpha:
                if len(vocab) >= VOCAB_SIZE:
                    break
                add_word_with_merges(c1 + c2, "subword")
                add_word_with_merges(" " + c1 + c2, "subword")
    if len(vocab) < VOCAB_SIZE:
        prefixes = [" ", "in", "re", "un", "de", "co", "st", "pr", "tr", "ch", "sh", "th", "wh", "py", "is", "to", "on", "at"]
        for pre in prefixes:
            for c1 in alpha:
                for c2 in alpha:
                    if len(vocab) >= VOCAB_SIZE:
                        break
                    add_word_with_merges(pre + c1 + c2, "subword")

    assert len(vocab) == VOCAB_SIZE, f"Expected {VOCAB_SIZE}, got {len(vocab)}"

    # Build HuggingFace Tokenizer
    bpe = models.BPE(vocab=vocab, merges=merges, unk_token="<|unk|>")
    tok = Tokenizer(bpe)
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.post_processor = processors.ByteLevel(trim_offsets=False)
    tok.add_special_tokens(SPECIAL_TOKENS)

    tok_path = OUT_DIR / "tokenizer.json"
    tok.save(str(tok_path))

    tok_cfg = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "bos_token": "<|bos|>",
        "eos_token": "<|eos|>",
        "pad_token": "<|pad|>",
        "unk_token": "<|unk|>",
        "model_max_length": 2048,
        "vocab_size": VOCAB_SIZE,
        "construction_method": "zero_training_programmatic_bpe",
    }
    (OUT_DIR / "tokenizer_config.json").write_text(json.dumps(tok_cfg, indent=2))

    # Save reverse vocab + categories for analytical weight synthesis
    id_to_token = {v: k for k, v in vocab.items()}
    meta = {
        "vocab_size": VOCAB_SIZE,
        "num_merges": len(merges),
        "categories": categories,
        "id_to_token": id_to_token,
    }
    (OUT_DIR / "token_metadata.json").write_text(json.dumps(meta))
    print(f"Built tokenizer.json: vocab_size={len(vocab)}, merges={len(merges)}")

    # Quick sanity check
    for sample in [
        "def analyze_data(arr):\n    return sum(arr) / len(arr)",
        "Hello! Can you analyze the time complexity of quicksort?",
    ]:
        enc = tok.encode(sample)
        dec = tok.decode(enc.ids)
        print(f"Sample ({len(enc.ids)} tokens): {enc.tokens[:12]} ... | roundtrip_ok={dec == sample}")


if __name__ == "__main__":
    build()
