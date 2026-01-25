#!/usr/bin/env python3
import os
import re
import sys
from dataclasses import dataclass
from typing import Any

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

def _try_parse_float(token: str) -> float | None:
    t = token.strip()
    if not t:
        return None
    try:
        # avoid treating plain ints as floats
        if re.fullmatch(r"[+-]?\d+", t):
            return None
        return float(t)
    except ValueError:
        return None

def _parse_default_value(raw: str) -> Any:
    if raw is None:
        return None
    s = raw.strip()
    if not s:
        return ""

    if s in ("true", "false"):
        return (s == "true")

    parsed_int = _try_parse_int(s)
    if parsed_int is not None:
        return parsed_int

    parsed_float = _try_parse_float(s)
    if parsed_float is not None:
        return parsed_float

    if (len(s) >= 2) and (
        (s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")
    ):
        return s[1:-1]

    return s

def _parse_enum_entries(enum_body: str) -> list[EnumEntry]:
    items: list[EnumEntry] = []
    next_value = 0

    raw_items = enum_body.split(",")
    for raw_item in raw_items:
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
    enum_pattern = re.compile(
        r"^\s*enum\s+class\s+(\w+)(?:\s*:\s*[\w:]+)?\s*\{([^}]*)\}\s*;\s*$",
        re.MULTILINE,
    )

    enum_map: dict[str, list[EnumEntry]] = {}
    def _strip_and_collect(match: re.Match) -> str:
        enum_name = match.group(1)
        enum_body = match.group(2)
        enum_entries = _parse_enum_entries(enum_body)
        enum_map[enum_name] = enum_entries
        return ""

    stripped_body = enum_pattern.sub(_strip_and_collect, struct_body)
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

    # Accept:
    #   EnumType::Value
    #   Namespace::EnumType::Value
    #   ...and also just "Value"
    m = re.fullmatch(r"(?:[\w:]+::)*(\w+)(?:::)?(\w+)", s)
    if m:
        tail0 = m.group(1)
        tail1 = m.group(2)
        enum_name_candidate = tail0
        enumerator_candidate = tail1

        if enum_name_candidate == enum_simple_name:
            for e in enum_entries:
                if e.name == enumerator_candidate:
                    return e.value

    for e in enum_entries:
        if e.name == s:
            return e.value

    return None

def generate_adapter_class(struct_name: str, members: list[Member]) -> str:
    adapter_name = f"{struct_name}Adapter"
    display_name = format_struct_name(struct_name)

    any_enums = any(m.is_enum for m in members)
    any_float3 = any(is_float3_type(m.type) for m in members)
    float3_indices: list[int] = [i for i, m in enumerate(members) if is_float3_type(m.type)]

    out: list[str] = []

    out.append(f"class {adapter_name} : public ConfigAdapter {{")
    out.append("    Q_OBJECT")
    out.append("public:")
    out.append(
        f"    explicit {adapter_name}({struct_name} &config, pc::devices::DevicePlugin* plugin, QObject *parent = nullptr)"
    )
    out.append("        : ConfigAdapter(plugin, parent), m_config(config) {}")
    out.append("")
    out.append(f"    QString configType() const override {{ return QStringLiteral(\"{struct_name}\"); }}")
    out.append(f"    Q_INVOKABLE QString displayName() const override {{ return QStringLiteral(\"{display_name}\"); }}")
    out.append("")
    out.append(f"    Q_INVOKABLE int fieldCount() const override {{ return {len(members)}; }}")
    out.append("")

    out.append("    void setConfigFromCore(const pc::devices::DeviceConfigurationVariant &v) override {")
    out.append(f"        const auto *cfg = std::get_if<{struct_name}>(&v);")
    out.append("        if (!cfg) return;")
    out.append("        m_config = *cfg;")
    out.append("        // Cheap + safe: notify all fields. Optimise later if needed.")
    out.append("        for (int i = 0; i < fieldCount(); ++i) {")
    out.append("            notifyFieldChanged(i);")
    out.append("        }")
    out.append("    }")
    out.append("")

    # float3 setters
    if any_float3:
        out.append("    // Dedicated setters for float3 fields (avoids QVariant conversion ambiguity from QML).")
        for i in float3_indices:
            member = members[i]
            out.append(
                f"    Q_INVOKABLE void set_{member.name}(float x, float y, float z) {{"
                f" emit fieldEditRequested({i}, QVariant::fromValue(QVector3D(x, y, z))); }}"
            )
        out.append("    Q_INVOKABLE void setFloat3Field(int index, float x, float y, float z) {")
        out.append("        emit fieldEditRequested(index, QVariant::fromValue(QVector3D(x, y, z)));")
        out.append("    }")
        out.append("")

    out.append("    Q_INVOKABLE QString fieldName(int index) const override {")
    out.append("        switch (index) {")
    for i, m in enumerate(members):
        out.append(f"        case {i}: return QStringLiteral(\"{m.name}\");")
    out.append("        default: return {};")
    out.append("        }")
    out.append("    }")
    out.append("")

    out.append("    Q_INVOKABLE QString fieldTypeName(int index) const override {")
    out.append("        switch (index) {")
    for i, m in enumerate(members):
        out.append(f"        case {i}: return QStringLiteral(\"{m.type}\");")
    out.append("        default: return {};")
    out.append("        }")
    out.append("    }")
    out.append("")

    if any_enums:
        out.append("    Q_INVOKABLE bool isEnum(int index) const override {")
        out.append("        switch (index) {")
        for i, m in enumerate(members):
            if m.is_enum:
                out.append(f"        case {i}: return true;")
        out.append("        default: return false;")
        out.append("        }")
        out.append("    }")
        out.append("")

        out.append("    Q_INVOKABLE QVariant enumOptions(int index) const override {")
        out.append("        switch (index) {")
        for i, m in enumerate(members):
            if not m.is_enum:
                continue
            out.append(f"        case {i}:")
            out.append("            return QVariantList{")
            for j, e in enumerate(m.enum_entries):
                comma = "," if j + 1 < len(m.enum_entries) else ""
                out.append(
                    f"                QVariantMap{{ {{\"text\", QStringLiteral(\"{e.name}\")}}, {{\"value\", {e.value}}} }}{comma}"
                )
            out.append("            };")
        out.append("        default: return {};")
        out.append("        }")
    out.append("    }")
    out.append("")

    out.append("    Q_INVOKABLE QVariant defaultValue(int index) const override {")
    out.append("        switch (index) {")
    for i, m in enumerate(members):
        v = m.default_value
        if v is None:
            out.append(f"        case {i}: return QVariant();")
        elif m.is_enum:
            enum_default = _enum_default_to_int(v, m.type, m.enum_entries)
            if enum_default is None and m.enum_entries:
                enum_default = m.enum_entries[0].value
            if enum_default is None:
                out.append(f"        case {i}: return QVariant();")
            else:
                out.append(f"        case {i}: return QVariant({enum_default});")
        elif is_float3_type(m.type):
            out.append(f"        case {i}: return QVariant::fromValue(QVector3D(0.0f, 0.0f, 0.0f));")
        elif isinstance(v, bool):
            out.append(f"        case {i}: return QVariant({str(v).lower()});")
        elif isinstance(v, int):
            out.append(f"        case {i}: return QVariant({v});")
        elif isinstance(v, float):
            out.append(f"        case {i}: return QVariant({v});")
        else:
            s = str(v).replace("\\", "\\\\").replace("\"", "\\\"")
            out.append(f"        case {i}: return QVariant(QStringLiteral(\"{s}\"));")
    out.append("        default: return {};")
    out.append("        }")
    out.append("    }")
    out.append("")

    out.append("    Q_INVOKABLE QVariant minMax(int index) const override {")
    out.append("        switch (index) {")
    for i, m in enumerate(members):
        if m.min_max:
            parts = [p.strip() for p in m.min_max.split(",")]
            mn = parts[0] if len(parts) >= 1 and parts[0] else "0"
            mx = parts[1] if len(parts) >= 2 and parts[1] else mn
            out.append(f"        case {i}: return QVariantList{{ QVariant({mn}), QVariant({mx}) }};")
        else:
            out.append(f"        case {i}: return {{}};")
    out.append("        default: return {};")
    out.append("        }")
    out.append("    }")
    out.append("")

    out.append("    Q_INVOKABLE bool isOptional(int index) const override {")
    out.append("        switch (index) {")
    for i, m in enumerate(members):
        out.append(f"        case {i}: return {'true' if m.optional else 'false'};")
    out.append("        default: return false;")
    out.append("        }")
    out.append("    }")
    out.append("")
    out.append("    Q_INVOKABLE bool isDisabled(int index) const override {")
    out.append("        switch (index) {")
    for i, m in enumerate(members):
        out.append(f"        case {i}: return {'true' if m.disabled else 'false'};")
    out.append("        default: return false;")
    out.append("        }")
    out.append("    }")
    out.append("")
    out.append("    Q_INVOKABLE bool isHidden(int index) const override {")
    out.append("        switch (index) {")
    for i, m in enumerate(members):
        out.append(f"        case {i}: return {'true' if m.hidden else 'false'};")
    out.append("        default: return false;")
    out.append("        }")
    out.append("    }")
    out.append("")

    out.append("    Q_INVOKABLE QVariant fieldValue(int index) const override {")
    out.append("        switch (index) {")
    for i, m in enumerate(members):
        if m.type == "std::string":
            out.append(f"        case {i}: return QVariant(QString::fromStdString(m_config.{m.name}));")
        elif m.is_enum:
            out.append(f"        case {i}: return QVariant(int(m_config.{m.name}));")
        elif is_float3_type(m.type):
            out.append(f"        case {i}: {{")
            out.append(f"            const auto &v = m_config.{m.name};")
            out.append("            return QVariant::fromValue(QVector3D(v.x, v.y, v.z));")
            out.append("        }")
        else:
            out.append(f"        case {i}: return QVariant::fromValue(m_config.{m.name});")
    out.append("        default: return {};")
    out.append("        }")
    out.append("    }")
    out.append("")

    # generated applyFieldValue(...) switch (type-specific mutation), no signals
    if any_float3:
        out.append("private:")
        out.append("    static QVector3D toQVector3DLoose(const QVariant &value) {")
        out.append("        if (value.canConvert<QVector3D>())")
        out.append("            return value.value<QVector3D>();")
        out.append("        if (value.metaType().id() == QMetaType::QVariantMap) {")
        out.append("            const auto m = value.toMap();")
        out.append("            return QVector3D(")
        out.append("                m.value(QStringLiteral(\"x\")).toFloat(),")
        out.append("                m.value(QStringLiteral(\"y\")).toFloat(),")
        out.append("                m.value(QStringLiteral(\"z\")).toFloat());")
        out.append("        }")
        out.append("        if (value.metaType().id() == QMetaType::QVariantList) {")
        out.append("            const auto l = value.toList();")
        out.append("            if (l.size() >= 3)")
        out.append("                return QVector3D(l[0].toFloat(), l[1].toFloat(), l[2].toFloat());")
        out.append("        }")
        out.append("        return QVector3D();")
        out.append("    }")
        out.append("")
        out.append("public:")

    out.append("    bool applyFieldValue(int index, const QVariant &value) override {")
    out.append("        switch (index) {")
    for i, m in enumerate(members):
        out.append(f"        case {i}: {{")
        if m.type == "std::string":
            out.append("            const std::string newValue = value.toString().toStdString();")
            out.append(f"            if (m_config.{m.name} == newValue)")
            out.append("                return false;")
            out.append(f"            m_config.{m.name} = newValue;")
            out.append("            return true;")
        elif m.is_enum:
            out.append("            const int newValueInt = value.toInt();")
            out.append(f"            const auto newValue = static_cast<{m.enum_qualified_type}>(newValueInt);")
            out.append(f"            if (m_config.{m.name} == newValue)")
            out.append("                return false;")
            out.append(f"            m_config.{m.name} = newValue;")
            out.append("            return true;")
        elif is_float3_type(m.type):
            out.append("            const QVector3D v = toQVector3DLoose(value);")
            out.append(f"            {m.type} newValue{{ v.x(), v.y(), v.z() }};")
            out.append(f"            if (m_config.{m.name} == newValue)")
            out.append("                return false;")
            out.append(f"            m_config.{m.name} = newValue;")
            out.append("            return true;")
        else:
            out.append(f"            const auto newValue = qvariant_cast<{m.type}>(value);")
            out.append(f"            if (m_config.{m.name} == newValue)")
            out.append("                return false;")
            out.append(f"            m_config.{m.name} = newValue;")
            out.append("            return true;")
        out.append("        }")
    out.append("        default:")
    out.append("            return false;")
    out.append("        }")
    out.append("    }")
    out.append("")

    # request-only setFieldValue
    out.append("    Q_INVOKABLE void setFieldValue(int index, const QVariant &value) override {")
    out.append("        emit fieldEditRequested(index, value);")
    out.append("    }")
    out.append("")

    out.append("private:")
    out.append(f"    {struct_name} &m_config;")
    out.append("};")

    return "\n".join(out)


def process_cpp_header(input_text: str, file_path: str) -> str:
    structs = re.findall(
        r"struct (\w+(?:Configuration|Workspace|Entry|Session|SessionLayout|BindingTarget|putRoute|putMapping|putChangeDetection|ABB)) {([\s\S]*?^\};)",
        input_text,
        re.MULTILINE,
    )

    type_pattern = r"([\w:]+(?:\:\:)?[\w:]+(?:<[\w\s:,]+>)?)"
    name_pattern = r"(\w+)"
    initial_value_pattern = r"(?:\s*=\s*([^;{]+)|\s*{\s*([^}]+)\s*})?"
    comment_pattern = r"(?:\s*//\s*(.*))?"
    minmax_pattern = re.compile(r"@minmax\(([^)]+)\)")
    optional_pattern = re.compile(r"@optional")
    disabled_pattern = re.compile(r"@disabled")
    hidden_pattern = re.compile(r"@hidden")
    verbatim_pattern = re.compile(
        r"^\s*((?:static|constexpr|inline|virtual|extern|using)\b.*)$",
        re.MULTILINE,
    )

    member_pattern = re.compile(
        fr"{type_pattern}\s+{name_pattern}{initial_value_pattern}\s*;{comment_pattern}",
        re.VERBOSE,
    )

    adapter_classes: list[str] = []
    needs_qvector3d = False

    for struct_name, struct_body in structs:
        nested_enums, stripped_struct_body = _extract_nested_enums(struct_body)

        members_body = verbatim_pattern.sub("", stripped_struct_body)

        unparsed_members = member_pattern.findall(members_body)
        members: list[Member] = []

        for member in unparsed_members:
            raw_type = member[0].strip()
            raw_name = member[1].strip()

            if is_float3_type(raw_type):
                needs_qvector3d = True

            init_value_raw = (member[2] or "").strip() or (member[3] or "").strip() or ""
            init_value_raw = init_value_raw.replace("{", "").replace("}", "").strip()
            init_value = _parse_default_value(init_value_raw)

            comment = member[4].strip() if member[4] else ""
            minmax_match = minmax_pattern.search(comment)

            is_enum = raw_type in nested_enums
            enum_entries = nested_enums.get(raw_type, [])
            enum_qualified_type = f"{struct_name}::{raw_type}" if is_enum else ""

            members.append(
                Member(
                    type=raw_type,
                    name=raw_name,
                    default_value=init_value,
                    min_max=minmax_match.group(1) if minmax_match else "",
                    optional=bool(optional_pattern.search(comment)),
                    disabled=bool(disabled_pattern.search(comment)),
                    hidden=bool(hidden_pattern.search(comment)),
                    is_enum=is_enum,
                    enum_qualified_type=enum_qualified_type,
                    enum_entries=enum_entries,
                )
            )

        adapter_classes.append(generate_adapter_class(struct_name, members))

    header_basename = os.path.basename(file_path)
    lines: list[str] = []
    lines.append("// auto-generated Qt/QML config adapters")
    lines.append("#pragma once")
    lines.append("")
    lines.append("#include <QObject>")
    lines.append("#include <QVariant>")
    lines.append("#include <QVariantMap>")
    lines.append("#include <QString>")
    lines.append("#include <QMetaType>")
    if needs_qvector3d:
        lines.append("#include <QVector3D>")
    lines.append("#include <variant>")
    lines.append("")
    lines.append('#include "models/config_adapter.h"')
    lines.append("#include <plugins/devices/device_variants.h>")
    lines.append(f'#include "{header_basename}"')
    lines.append("")

    ns_match = re.search(r"namespace\s+([A-Za-z_][\w:]*)\s*{", input_text)

    if adapter_classes:
        if ns_match:
            ns = ns_match.group(1)
            lines.append(f"namespace {ns} {{")
            lines.append("")
            lines.append("\n\n".join(adapter_classes))
            lines.append("")
            lines.append(f"}} // namespace {ns}")
        else:
            lines.append("\n\n".join(adapter_classes))
    else:
        lines.append("// no matching config structs found in this header")

    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate-qt-adapters.py <filename1> <filename2> ...")
        sys.exit(1)

    for file_name in sys.argv[1:]:
        with open(file_name, "r", encoding="utf-8") as input_file:
            file_content = input_file.read()
            generated_content = process_cpp_header(file_content, file_name)
            generated_file_name = file_name.replace(".h", ".gen.h")
            with open(generated_file_name, "w", encoding="utf-8") as output_file:
                output_file.write(generated_content)
            print("--", generated_file_name)
