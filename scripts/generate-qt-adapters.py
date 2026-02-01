#!/usr/bin/env python3
import os
import re
import sys
from dataclasses import dataclass
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

# ----------------------------
# Data model
# ----------------------------

@dataclass
class EnumEntry:
    name: str
    value: int


@dataclass
class Member:
    type: str
    name: str
    default_value: Any
    min_max: str
    optional: bool
    disabled: bool
    hidden: bool
    is_enum: bool
    enum_qualified_type: str
    enum_entries: list[EnumEntry]


@dataclass(frozen=True, slots=True)
class GeneratorArgs:
    templates_dir: str
    out_dir: str
    src_root: str
    input_headers: list[str]


# ----------------------------
# regexes necessary for multiple funcs are compiled once up here
# ----------------------------

STRUCTS_RE = re.compile(
    r"struct (\w+(?:Configuration|Workspace|Entry|Session|SessionLayout|BindingTarget|putRoute|putMapping|putChangeDetection|ABB)) {([\s\S]*?^\};)",
    re.MULTILINE,
)

TYPE_PATTERN = r"([\w:]+(?:\:\:)?[\w:]+(?:<[\w\s:,]+>)?)"
NAME_PATTERN = r"(\w+)"
INITIAL_VALUE_PATTERN = r"(?:\s*=\s*([^;{]+)|\s*{\s*([^}]+)\s*})?"
COMMENT_PATTERN = r"(?:\s*//\s*(.*))?"

MEMBER_RE = re.compile(
    fr"{TYPE_PATTERN}\s+{NAME_PATTERN}{INITIAL_VALUE_PATTERN}\s*;{COMMENT_PATTERN}",
    re.VERBOSE,
)

MINMAX_RE = re.compile(r"@minmax\(([^)]+)\)")
OPTIONAL_RE = re.compile(r"@optional")
DISABLED_RE = re.compile(r"@disabled")
HIDDEN_RE = re.compile(r"@hidden")
VERBATIM_RE = re.compile(
    r"^\s*((?:static|constexpr|inline|virtual|extern|using)\b.*)$",
    re.MULTILINE,
)

NAMESPACE_RE = re.compile(r"namespace\s+([A-Za-z_][\w:]*)\s*{")

ENUM_DECL_RE = re.compile(
    r"^\s*enum\s+class\s+(\w+)(?:\s*:\s*[\w:]+)?\s*\{([^}]*)\}\s*;\s*$",
    re.MULTILINE,
)

ENUM_DEFAULT_RE = re.compile(r"(?:[\w:]+::)*(\w+)(?:::)?(\w+)")
INTLIKE_RE = re.compile(r"(u?)int(8|16|32|64)_t")
OPTIONAL_T_RE = re.compile(r"std::optional<\s*([^>]+)\s*>")


# ----------------------------
# Helper funcs
# ----------------------------

def is_float3_type(type_name: str) -> bool:
    t = type_name.strip()
    return t in ("pc::float3", "float3")


def format_struct_name(name: str) -> str:
    name = name.replace("Configuration", "")
    name = name.replace("Operator", "")
    result: list[str] = []
    for i, char in enumerate(name):
        if i > 0 and char.isupper() and name[i - 1].islower():
            result.append(" ")
        result.append(char)
    return "".join(result)


def _try_parse_int(token: str) -> int | None:
    t = token.strip()
    if not t:
        return None
    try:
        return int(t, 0)
    except ValueError:
        return None


def _parse_default_value(raw: str) -> Any:
    if raw is None:
        return None
    s = raw.strip()
    if not s:
        return ""

    if s in ("true", "false"):
        return s == "true"

    parsed_int = _try_parse_int(s)
    if parsed_int is not None:
        return parsed_int

    try:
        parsed_float = float(s)
    except ValueError:
        parsed_float = None

    if parsed_float is not None and not isinstance(parsed_float, bool):
        return parsed_float

    if (len(s) >= 2) and (
        (s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")
    ):
        return s[1:-1]

    return s


def _parse_enum_entries(enum_body: str) -> list[EnumEntry]:
    items: list[EnumEntry] = []
    next_value = 0

    for raw_item in enum_body.split(","):
        entry = raw_item.strip()
        if not entry:
            continue

        entry = re.sub(r"//.*$", "", entry).strip()
        entry = re.sub(r"/\*.*?\*/", "", entry).strip()
        if not entry:
            continue

        if "=" in entry:
            name_part, value_part = entry.split("=", 1)
            enum_name = name_part.strip()
            explicit_value = _try_parse_int(value_part)
            if explicit_value is None:
                continue
            items.append(EnumEntry(name=enum_name, value=explicit_value))
            next_value = explicit_value + 1
        else:
            items.append(EnumEntry(name=entry, value=next_value))
            next_value += 1

    return items


def _extract_nested_enums(struct_body: str) -> tuple[dict[str, list[EnumEntry]], str]:
    enum_map: dict[str, list[EnumEntry]] = {}

    def _strip_and_collect(match: re.Match) -> str:
        enum_name = match.group(1)
        enum_body = match.group(2)
        enum_map[enum_name] = _parse_enum_entries(enum_body)
        return ""

    stripped_body = ENUM_DECL_RE.sub(_strip_and_collect, struct_body)
    return enum_map, stripped_body


def _enum_default_to_int(
    default_value: Any, enum_simple_name: str, enum_entries: list[EnumEntry]
) -> int | None:
    if default_value is None:
        return None
    if isinstance(default_value, int):
        return default_value
    s = str(default_value).strip()
    if not s:
        return None

    m = ENUM_DEFAULT_RE.fullmatch(s)
    if m:
        enum_name_candidate = m.group(1)
        enumerator_candidate = m.group(2)
        if enum_name_candidate == enum_simple_name:
            for e in enum_entries:
                if e.name == enumerator_candidate:
                    return e.value

    for e in enum_entries:
        if e.name == s:
            return e.value

    return None


def _is_simple_comparable_type(type_name: str) -> bool:
    t = type_name.strip()

    if t in ("bool", "int", "unsigned", "float", "double", "std::string", "QString"):
        return True

    if is_float3_type(t):
        return True

    if INTLIKE_RE.fullmatch(t):
        return True

    m = OPTIONAL_T_RE.fullmatch(t)
    if m:
        inner = m.group(1).strip()
        return _is_simple_comparable_type(inner)

    return False


def default_value_case_cpp(member: Member) -> str:
    v = member.default_value

    if v is None:
        return "return QVariant();"

    if member.is_enum:
        enum_default = _enum_default_to_int(v, member.type, member.enum_entries)
        if enum_default is None and member.enum_entries:
            enum_default = member.enum_entries[0].value
        if enum_default is None:
            return "return QVariant();"
        return f"return QVariant({enum_default});"

    if is_float3_type(member.type):
        return "return QVariant::fromValue(QVector3D(0.0f, 0.0f, 0.0f));"

    if isinstance(v, bool):
        return f"return QVariant({'true' if v else 'false'});"

    if isinstance(v, int):
        return f"return QVariant({v});"

    if isinstance(v, float):
        return f"return QVariant({v});"

    s = str(v).replace("\\", "\\\\").replace('"', '\\"')
    return f'return QVariant(QStringLiteral("{s}"));'


def minmax_case_cpp(member: Member) -> str:
    if member.min_max:
        parts = [p.strip() for p in member.min_max.split(",")]
        min_value = parts[0] if len(parts) >= 1 and parts[0] else "0"
        max_value = parts[1] if len(parts) >= 2 and parts[1] else min_value
        return f"return QVariantList{{ QVariant({min_value}), QVariant({max_value}) }};"
    return "return {};"


# ----------------------------
# Path helpers
# ----------------------------

def _rel_header_path(file_path: str, src_root: str) -> str:
    input_abs = os.path.abspath(file_path)
    src_root_abs = os.path.abspath(src_root)
    rel = os.path.relpath(input_abs, src_root_abs).replace("\\", "/")
    if rel.startswith("src/"):
        rel = rel[len("src/") :]
    return rel


def _generated_adapter_include_for_header_rel(rel_header: str) -> str:
    """
    Given 'camera/camera_config.h' -> 'camera/camera_config_adapter.gen.h'
    Given 'plugins/devices/orbbec/orbbec_device_config.h' -> 'plugins/devices/orbbec/orbbec_device_adapter.gen.h'
    """
    rel_dir = os.path.dirname(rel_header)
    base = os.path.basename(rel_header)
    base_no_ext = re.sub(r"\.h$", "", base)

    is_device_header = os.path.basename(rel_header).endswith("_device_config.h")
    stem = base_no_ext
    if is_device_header and base_no_ext.endswith("_device_config"):
        stem = base_no_ext[: -len("_config")]

    return os.path.join(rel_dir, f"{stem}_adapter.gen.h").replace("\\", "/")


def _scan_structs_in_header(input_text: str) -> list[str]:
    return [struct_name for struct_name, _ in STRUCTS_RE.findall(input_text)]


# ----------------------------
# Flattened path computation
# ----------------------------

def _is_nested_config_member(member: Member) -> bool:
    """
    "Nested" means: endswith Configuration AND has an adapter generated (i.e. not a scalar/string/bool/enum/float3/simple).
    """
    if not member.type.endswith("Configuration"):
        return False
    if member.type in ("std::string", "QString", "bool"):
        return False
    if member.is_enum:
        return False
    if is_float3_type(member.type) or _is_simple_comparable_type(member.type):
        return False
    return True


def compute_flattened_paths(
    struct_name: str,
    struct_members: list[Member],
    struct_members_map: dict[str, list[Member]],
) -> list[str]:
    """
    Returns leaf paths like:
      ["id", "label", "camera/locked", "camera/show_grid"]
    for the given struct.
    """

    def _recurse(current_struct_name: str, prefix: str, visiting: set[str]) -> list[str]:
        if current_struct_name in visiting:
            # break cycles (shouldn't happen in configs, but keep it safe)
            return []
        visiting.add(current_struct_name)

        members = struct_members_map.get(current_struct_name, [])
        out: list[str] = []

        for member in members:
            member_path = f"{prefix}{member.name}" if prefix else member.name

            if _is_nested_config_member(member) and (member.type in struct_members_map):
                # Expand nested config and prefix with "member/"
                nested_prefix = f"{member_path}/"
                out.extend(_recurse(member.type, nested_prefix, visiting))
            else:
                # Leaf (or unknown nested type): keep as leaf path
                out.append(member_path)

        visiting.remove(current_struct_name)
        return out

    # Ensure we use the exact members passed for the top-level struct even if the map differs.
    # (But usually they match.)
    struct_members_map = dict(struct_members_map)
    struct_members_map[struct_name] = struct_members

    return _recurse(struct_name, prefix="", visiting=set())


# ----------------------------
# Parsing helpers
# ----------------------------

def _parse_members_for_struct(struct_name: str, struct_body: str) -> tuple[list[Member], bool]:
    nested_enums, stripped_struct_body = _extract_nested_enums(struct_body)
    members_body = VERBATIM_RE.sub("", stripped_struct_body)

    needs_qvector3d = False
    members: list[Member] = []

    for raw_type, raw_name, init_eq, init_brace, raw_comment in MEMBER_RE.findall(members_body):
        raw_type = raw_type.strip()
        raw_name = raw_name.strip()

        if is_float3_type(raw_type):
            needs_qvector3d = True

        init_value_raw = (init_eq or "").strip() or (init_brace or "").strip() or ""
        init_value_raw = init_value_raw.replace("{", "").replace("}", "").strip()
        init_value = _parse_default_value(init_value_raw)

        comment = raw_comment.strip() if raw_comment else ""
        minmax_match = MINMAX_RE.search(comment)

        is_enum = raw_type in nested_enums
        enum_entries = nested_enums.get(raw_type, [])
        enum_qualified_type = f"{struct_name}::{raw_type}" if is_enum else ""

        members.append(
            Member(
                type=raw_type,
                name=raw_name,
                default_value=init_value,
                min_max=minmax_match.group(1) if minmax_match else "",
                optional=bool(OPTIONAL_RE.search(comment)),
                disabled=bool(DISABLED_RE.search(comment)),
                hidden=bool(HIDDEN_RE.search(comment)),
                is_enum=is_enum,
                enum_qualified_type=enum_qualified_type,
                enum_entries=enum_entries,
            )
        )

    return members, needs_qvector3d


# ----------------------------
# Main header generation
# ----------------------------

def process_cpp_header(
    input_text: str,
    file_path: str,
    env: Environment,
    src_root: str,
    struct_to_adapter_include: dict[str, str],
    struct_members_map: dict[str, list[Member]],
) -> str:
    structs = STRUCTS_RE.findall(input_text)

    ns_match = NAMESPACE_RE.search(input_text)
    namespace_name = ns_match.group(1) if ns_match else ""
    is_device_namespace = namespace_name == "pc::devices"

    needs_qvector3d = False
    rendered_structs: list[dict[str, Any]] = []

    for struct_name, struct_body in structs:
        members, struct_needs_qvector3d = _parse_members_for_struct(struct_name, struct_body)
        if struct_needs_qvector3d:
            needs_qvector3d = True

        # Adapter class name
        adapter_class_base = struct_name
        if is_device_namespace and struct_name.endswith("DeviceConfiguration"):
            adapter_class_base = struct_name[: -len("Configuration")]
        adapter_name = f"{adapter_class_base}Adapter"

        # Nested adapter includes:
        # - For core config adapters: include nested config adapter headers (same as before).
        # - For device adapters: nested device structures don't exist, but nested regular config structures can.
        nested_adapter_includes: list[str] = []
        for m in members:
            if not _is_nested_config_member(m):
                continue

            inc = struct_to_adapter_include.get(m.type)
            if inc and inc not in nested_adapter_includes:
                nested_adapter_includes.append(inc)

        flattened_paths = compute_flattened_paths(struct_name, members, struct_members_map)

        rendered_structs.append(
            {
                "struct_name": struct_name,
                "adapter_name": adapter_name,
                "label": format_struct_name(struct_name),
                "members": members,
                "any_enums": any(m.is_enum for m in members),
                "any_float3": any(is_float3_type(m.type) for m in members),
                "nested_adapter_includes": nested_adapter_includes,
                "flattened_paths": flattened_paths,
            }
        )

    rel = _rel_header_path(file_path, src_root)
    header_include_path = rel

    template_name = "device_adapter_impl.h.j2" if is_device_namespace else "config_adapter_impl.h.j2"
    template = env.get_template(template_name)

    return template.render(
        header_include_path=header_include_path,
        namespace_name=namespace_name,
        needs_qvector3d=needs_qvector3d,
        structs=rendered_structs,
    )


# ----------------------------
# entrypoint
# ----------------------------

def _parse_args(argv: list[str]) -> GeneratorArgs:
    templates_dir: str | None = None
    out_dir: str | None = None
    src_root: str | None = None
    input_headers: list[str] = []

    it = iter(argv)
    for tok in it:
        if tok in ("--templates-dir", "-t"):
            templates_dir = next(it, None)
        elif tok in ("--out-dir", "-o"):
            out_dir = next(it, None)
        elif tok == "--src-root":
            src_root = next(it, None)
        else:
            input_headers.append(tok)

    if not templates_dir or not out_dir or not src_root:
        print(
            "Usage: python generate-qt-adapters.py "
            "--templates-dir <dir> --out-dir <dir> --src-root <dir> <file1> <file2> ...",
            file=sys.stderr,
        )
        raise SystemExit(2)

    if not input_headers:
        print("error: no input headers provided", file=sys.stderr)
        raise SystemExit(2)

    return GeneratorArgs(
        templates_dir=os.path.abspath(templates_dir),
        out_dir=os.path.abspath(out_dir),
        src_root=os.path.abspath(src_root),
        input_headers=input_headers,
    )


def main() -> int:
    args = _parse_args(sys.argv[1:])

    env = Environment(
        loader=FileSystemLoader(args.templates_dir),
        undefined=StrictUndefined,
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )

    env.globals["is_float3_type"] = is_float3_type
    env.globals["is_simple_comparable_type"] = _is_simple_comparable_type
    env.globals["default_value_case_cpp"] = default_value_case_cpp
    env.globals["minmax_case_cpp"] = minmax_case_cpp

    # ---------- Pass 1: scan headers, cache text, build maps ----------
    header_cache: dict[str, str] = {}

    # Map: struct name -> generated adapter include path
    struct_to_adapter_include: dict[str, str] = {}

    # Map: struct name -> parsed members (used for flattened path recursion)
    struct_members_map: dict[str, list[Member]] = {}

    for file_name in sorted(args.input_headers):
        with open(file_name, "r", encoding="utf-8") as input_file:
            file_content = input_file.read()
        header_cache[file_name] = file_content

        rel_header = _rel_header_path(file_name, args.src_root)
        adapter_include = _generated_adapter_include_for_header_rel(rel_header)

        for struct_name, struct_body in STRUCTS_RE.findall(file_content):
            # struct->include mapping
            struct_to_adapter_include.setdefault(struct_name, adapter_include)

            # struct->members mapping (for recursion)
            members, _ = _parse_members_for_struct(struct_name, struct_body)
            struct_members_map.setdefault(struct_name, members)

    # ---------- Pass 2: render + write ----------
    for file_name in sorted(args.input_headers):
        file_content = header_cache[file_name]

        generated_content = process_cpp_header(
            file_content,
            file_name,
            env,
            args.src_root,
            struct_to_adapter_include,
            struct_members_map,
        )

        rel = _rel_header_path(file_name, args.src_root)
        rel_dir = os.path.dirname(rel)
        base = os.path.basename(rel)
        base_no_ext = re.sub(r"\.h$", "", base)

        is_device_header = os.path.basename(rel).endswith("_device_config.h")
        stem = base_no_ext
        if is_device_header and base_no_ext.endswith("_device_config"):
            stem = base_no_ext[: -len("_config")]

        generated_file_name = os.path.join(args.out_dir, rel_dir, f"{stem}_adapter.gen.h")
        os.makedirs(os.path.dirname(generated_file_name), exist_ok=True)

        with open(generated_file_name, "w", encoding="utf-8") as output_file:
            output_file.write(generated_content)

        print("--", generated_file_name)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())