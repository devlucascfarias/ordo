"""Esquemas de saída do Ordo, parsing, reparo de YAML e serialização segura dos alvos de treino.

Usado pelo CLI (validar a resposta), pelo benchmark (pontuar) e pela geração de dados (filtrar).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from .contract import GcMode

# ---------------------------------------------------------------- modelos


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TraceItem(_Strict):
    file: str
    lines: str
    symbol: Optional[str] = None
    issue: str

    @field_validator("lines", mode="before")
    @classmethod
    def _lines_to_str(cls, v: Any) -> Any:
        return str(v) if isinstance(v, int) else v


class Snippet(_Strict):
    location: str
    code: str

    @field_validator("code", mode="before")
    @classmethod
    def _unescape_newlines(cls, v: Any) -> Any:
        # o modelo publicado às vezes escreve '116 | a\n117 | b' entre aspas simples (\n literal)
        if isinstance(v, str) and "\n" not in v and "\\n" in v:
            return v.replace("\\n", "\n").replace('\\"', '"')
        return v


class ToolOutput(_Strict):
    # FOUND/NOT_FOUND: o modelo publicado usa em saídas de busca (`ordo search`), que passam por TOOL_OUTPUT
    status: Literal["PASS", "FAIL", "ERROR", "WARN", "FOUND", "NOT_FOUND"]
    summary: str
    causal_trace: list[TraceItem] = []
    exact_snippets: list[Snippet] = []
    dropped_noise: Optional[str] = None


class CodeSlice(_Strict):
    status: Literal["FOUND", "NOT_FOUND", "PARTIAL"]
    summary: str
    relevant_symbols: list[TraceItem] = []
    exact_snippets: list[Snippet] = []
    dropped_noise: Optional[str] = None


class DeadEnd(_Strict):
    attempt: str
    reason_failed: str


class FileState(_Strict):
    file: str
    test_status: str


class MemoryCompact(_Strict):
    status: Literal["COMPACTED"]
    active_goal: str
    confirmed_facts: list[str] = []
    dead_ends_do_not_repeat: list[DeadEnd] = []
    modified_files_state: list[FileState] = []


SCHEMAS: dict[GcMode, type[_Strict]] = {
    GcMode.TOOL_OUTPUT: ToolOutput,
    GcMode.CODE_SLICE: CodeSlice,
    GcMode.MEMORY_COMPACT: MemoryCompact,
}

# ---------------------------------------------------------------- reparo

_FENCE = re.compile(r"^\s*```[\w-]*\s*$")
_STRUCT = re.compile(r"^(?P<indent>\s*)(?P<dash>- )?(?P<key>[A-Za-z_][\w]*):(?:\s+(?P<value>.*))?$")
_LIST_ITEM = re.compile(r"^\s*- ")


def strip_fences(text: str) -> str:
    lines = [l for l in text.strip().splitlines() if not _FENCE.match(l)]
    return "\n".join(lines)


def _decode_single(inner: str) -> str:
    """Conteúdo entre aspas simples. Se há aspas simples soltas (o modelo não escapou), usa literal."""
    if "'" in inner.replace("''", ""):
        return inner
    return inner.replace("''", "'")


def _requote(value: str) -> str:
    """Reescreve um valor escalar como string YAML entre aspas duplas (JSON é YAML válido)."""
    v = value.strip()
    if len(v) >= 2 and v[0] == v[-1] == "'":
        return json.dumps(_decode_single(v[1:-1]), ensure_ascii=False)
    if len(v) >= 2 and v[0] == v[-1] == '"':
        try:
            return json.dumps(json.loads(v), ensure_ascii=False)
        except json.JSONDecodeError:
            return json.dumps(v[1:-1], ensure_ascii=False)
    if v.startswith("'") or v.startswith('"'):  # aberta e nunca fechada
        return json.dumps(v[1:], ensure_ascii=False)
    return json.dumps(v, ensure_ascii=False)


def repair_yaml_output(raw_text: str) -> str:
    """Conserta os defeitos típicos do modelo publicado, linha a linha.

    - remove cercas de markdown;
    - reescreve escalares com aspas simples internas não escapadas (PHP, Ruby, SQL);
    - junta linhas de continuação soltas ao valor anterior.
    Só reescreve valores; a estrutura (chaves, listas, indentação) fica como veio.
    """
    text = strip_fences(raw_text)
    groups: list[tuple[re.Match | None, str, list[str]]] = []
    for line in text.splitlines():
        m = _STRUCT.match(line)
        if m or not groups:
            groups.append((m, line, []))
        else:
            groups[-1][2].append(line)

    out: list[str] = []
    for m, line, cont in groups:
        while cont and not cont[-1].strip():
            cont.pop()
        value = m.group("value") if m else None
        if m is None or value is None or value.strip() in ("", "|", "|-", ">", ">-"):
            out.append(line)
            out.extend(cont)
            continue
        if not cont:
            try:
                ok = isinstance(yaml.safe_load(f"k: {value}"), dict)
            except yaml.YAMLError:
                ok = False
            out.append(line if ok else f"{line[: m.start('value')]}{_requote(value)}")
            continue
        joined = "\n".join([value] + [c.strip() for c in cont if c.strip()])
        out.append(f"{line[: m.start('value')]}{_requote(joined)}")
    return "\n".join(out)


# ---------------------------------------------------------------- parsing


@dataclass
class ParseResult:
    mode: GcMode
    raw_yaml_ok: bool  # o texto do modelo já era YAML (mapa) sem cercas
    repaired: bool  # precisou do repair para virar YAML
    data: Optional[dict] = None
    obj: Optional[_Strict] = None
    lenient: Optional[_Strict] = None  # válido só depois de descartar chaves extras de nível superior
    errors: list[str] = field(default_factory=list)

    @property
    def yaml_ok(self) -> bool:
        return self.data is not None

    @property
    def schema_ok(self) -> bool:
        return self.obj is not None


def _load_map(text: str) -> Optional[dict]:
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        return None
    return data if isinstance(data, dict) else None


def parse_output(text: str, mode: GcMode | str) -> ParseResult:
    mode = GcMode(mode)
    has_fence = any(_FENCE.match(l) for l in text.splitlines())
    data = None if has_fence else _load_map(text)
    res = ParseResult(mode=mode, raw_yaml_ok=data is not None, repaired=False)
    if data is None:
        data = _load_map(repair_yaml_output(text))
        res.repaired = data is not None
    if data is None:
        res.errors.append("yaml: não foi possível interpretar nem após o reparo")
        return res
    res.data = data
    try:
        res.obj = SCHEMAS[mode].model_validate(data)
    except ValidationError as e:
        for err in e.errors():
            res.errors.append(f"schema: {'.'.join(str(p) for p in err['loc'])}: {err['msg']}")
        # para verificar o conteúdo mesmo assim: ignora só chaves extras no nível de cima
        schema = SCHEMAS[mode]
        alt = dict(data)
        if mode is GcMode.CODE_SLICE and "relevant_symbols" not in alt and "causal_trace" in alt:
            alt["relevant_symbols"] = alt.pop("causal_trace")  # o modelo publicado usa a chave do TOOL_OUTPUT
        try:
            res.lenient = schema.model_validate({k: v for k, v in alt.items() if k in schema.model_fields})
        except ValidationError:
            pass
    return res


# ---------------------------------------------------------------- alvos de treino


class _TargetDumper(yaml.SafeDumper):
    pass


def _str_repr(dumper: yaml.SafeDumper, value: str) -> yaml.Node:
    style = "|" if "\n" in value and not any(l != l.rstrip() for l in value.splitlines()) else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)


_TargetDumper.add_representer(str, _str_repr)


def dump_target(obj: _Strict | dict) -> str:
    """Serializa um alvo de treino como YAML seguro (D6): nunca escrito à mão, sempre reparseável."""
    data = obj.model_dump(exclude_none=True) if isinstance(obj, BaseModel) else obj
    if isinstance(obj, BaseModel):  # listas vazias não são escritas (saída mais curta; regra 4 no PASS)
        data = {k: v for k, v in data.items() if v != []}
    text = yaml.dump(data, Dumper=_TargetDumper, sort_keys=False, allow_unicode=True, width=10**9)
    assert yaml.safe_load(text) == data, "dump_target não é reversível"
    return text.rstrip("\n")
