#!/usr/bin/env python3
import os
import re
import sys
from dataclasses import dataclass

@dataclass
class Member:
    type: str
    name: str
    default_value: str
    min_max: str
    optional: bool
    disabled: bool
    hidden: bool

file_cache = {}

def format_struct_name(name):
    name = name.replace("Configuration", "")
    name = name.replace("Operator", "")
    result = []
    for i, char in enumerate(name):
        if i > 0 and char.isupper() and name[i - 1].islower():
            result.append(' ')
        result.append(char)
    return ''.join(result)

# still here if you ever want it again for other generators
def calculate_relative_path(from_file, to_file):
    from_dir = os.path.dirname(from_file)
    relative_path = os.path.relpath(to_file, from_dir)
    return relative_path

def ensure_headers(file_content, header_paths):
    lines = file_content.splitlines()

    # find the index of the last #include
    last_include_index = next(
        (i for i, line in reversed(list(enumerate(lines))) if line.startswith('#include')),
        -1
    )

    # format the header paths into includes
    headers_to_insert = [
        f'#include "{header}"'
        for header in header_paths
        if f'#include "{header}"' not in file_content
    ]

    # insert the headers after the last #include
    if headers_to_insert:
        insert_index = last_include_index + 1 if last_include_index != -1 else 0
        lines[insert_index:insert_index] = headers_to_insert

    return '\n'.join(lines)

def generate_adapter_class(struct_name: str, members: list[Member]) -> str:
    """
    Given a single config struct and its members, spit out the corresponding
    QObject-derived ConfigAdapter class that wraps a reference to that struct.
    """

    adapter_name = f"{struct_name}Adapter"
    display_name = format_struct_name(struct_name)

    out: list[str] = []

    out.append(f"class {adapter_name} : public ConfigAdapter {{")
    out.append("    Q_OBJECT")
    out.append("public:")
    out.append(f"    explicit {adapter_name}({struct_name} &config, pc::devices::DevicePlugin* plugin, QObject *parent = nullptr)")
    out.append(f"        : ConfigAdapter(plugin, parent), m_config(config) {{}}")
    out.append("")
    out.append(f"    QString configType() const override {{ return QStringLiteral(\"{struct_name}\"); }}")
    out.append(f"    Q_INVOKABLE QString displayName() const override {{ return QStringLiteral(\"{display_name}\"); }}")
    out.append("")
    out.append(f"    Q_INVOKABLE int fieldCount() const override {{ return {len(members)}; }}")
    out.append("")

    # fieldName()
    out.append("    Q_INVOKABLE QString fieldName(int index) const override {")
    out.append("        switch (index) {")
    for i, m in enumerate(members):
        out.append(f"        case {i}: return QStringLiteral(\"{m.name}\");")
    out.append("        default: return {};")
    out.append("        }")
    out.append("    }")
    out.append("")

    # fieldTypeName()
    out.append("    Q_INVOKABLE QString fieldTypeName(int index) const override {")
    out.append("        switch (index) {")
    for i, m in enumerate(members):
        out.append(f"        case {i}: return QStringLiteral(\"{m.type}\");")
    out.append("        default: return {};")
    out.append("        }")
    out.append("    }")
    out.append("")

    # defaultValue()
    out.append("    Q_INVOKABLE QVariant defaultValue(int index) const override {")
    out.append("        switch (index) {")
    for i, m in enumerate(members):
        v = m.default_value
        if v is None:
            out.append(f"        case {i}: return QVariant();")
        elif isinstance(v, bool):
            out.append(f"        case {i}: return QVariant({str(v).lower()});")
        elif isinstance(v, int) or isinstance(v, float):
            out.append(f"        case {i}: return QVariant({v});")
        else:
            s = str(v).replace("\\", "\\\\").replace("\"", "\\\"")
            out.append(f"        case {i}: return QVariant(QStringLiteral(\"{s}\"));")
    out.append("        default: return {};")
    out.append("        }")
    out.append("    }")
    out.append("")

    # minMax() returns a QVariantList of 2 numbers (as QVariants) or an empty QVariant
    out.append("    Q_INVOKABLE QVariant minMax(int index) const override {")
    out.append("        switch (index) {")
    for i, m in enumerate(members):
        if m.min_max:
            # m.min_max is the raw "0.0, 1.0" or "0, 1" from @minmax(...)
            parts = [p.strip() for p in m.min_max.split(",")]
            # fallbacks just in case the annotation is weird
            mn = parts[0] if len(parts) >= 1 and parts[0] else "0"
            mx = parts[1] if len(parts) >= 2 and parts[1] else mn
            # we explicitly wrap in QVariant so the QList<QVariant> initializer-list is valid
            out.append(
                f"        case {i}: return QVariantList{{ QVariant({mn}), QVariant({mx}) }};"
            )
        else:
            out.append(f"        case {i}: return {{}};")
    out.append("        default: return {};")
    out.append("        }")
    out.append("    }")
    out.append("")

    # optional / disabled / hidden flags
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

    # fieldValue()
    out.append("    Q_INVOKABLE QVariant fieldValue(int index) const override {")
    out.append("        switch (index) {")
    for i, m in enumerate(members):
        if m.type == "std::string":
            # expose as QString to QML
            out.append(
                f"        case {i}: return QVariant(QString::fromStdString(m_config.{m.name}));"
            )
        else:
            out.append(
                f"        case {i}: return QVariant::fromValue(m_config.{m.name});"
            )
    out.append("        default: return {};")
    out.append("        }")
    out.append("    }")
    out.append("")

    # setFieldValue()
    out.append("    Q_INVOKABLE void setFieldValue(int index, const QVariant &value) override {")
    out.append("        switch (index) {")
    for i, m in enumerate(members):
        out.append(f"        case {i}: {{")
        if m.type == "std::string":
            # QML gives us a string -> QString -> std::string
            out.append("            auto newValue = value.toString().toStdString();")
        else:
            out.append(f"            auto newValue = qvariant_cast<{m.type}>(value);")
        out.append(f"            if (m_config.{m.name} == newValue)")
        out.append("                return;")
        out.append(f"            m_config.{m.name} = newValue;")
        out.append("            emit fieldChanged(index);")
        out.append("            break;")
        out.append("        }")
    out.append("        default:")
    out.append("            break;")
    out.append("        }")
    out.append("    }")
    out.append("")

    out.append("private:")
    out.append(f"    {struct_name} &m_config;")
    out.append("};")

    return "\n".join(out)

def process_cpp_header(input_text, file_path):
    # TODO: replace this explicit string search with a search for a meta comment
    # which instructs to serialize the struct
    structs = re.findall(
        r"struct (\w+(?:Configuration|Workspace|Entry|Session|SessionLayout|BindingTarget|putRoute|putMapping|putChangeDetection|ABB)) {([\s\S]*?^\};)",
        input_text,
        re.MULTILINE,
    )

    # define regex components
    type_pattern = r"([\w:]+(?:\:\:)?[\w:]+(?:<[\w\s:,]+>)?)"
    name_pattern = r"(\w+)"
    initial_value_pattern = r"(?:\s*=\s*([^;{]+)|\s*{\s*([^}]+)\s*})?"
    comment_pattern = r"(?:\s*//\s*(.*))?"
    minmax_pattern = re.compile(r"@minmax\(([^)]+)\)")
    optional_pattern = re.compile(r"@optional")
    disabled_pattern = re.compile(r"@disabled")
    hidden_pattern = re.compile(r"@hidden")
    verbatim_pattern = re.compile(
        r'^\s*((?:static|constexpr|inline|virtual|extern|using)\b.*)$',
        re.MULTILINE,
    )

    member_pattern = re.compile(
        fr"{type_pattern}\s+{name_pattern}{initial_value_pattern}\s*;{comment_pattern}",
        re.VERBOSE,
    )

    adapter_classes: list[str] = []

    for struct_name, struct_body in structs:
        # strip out static/constexpr/using/etc – same as before
        verbatim_members = verbatim_pattern.findall(struct_body)
        members_body = verbatim_pattern.sub('', struct_body)

        # now find simple member variables
        unparsed_members = member_pattern.findall(members_body)
        members: list[Member] = []

        for member in unparsed_members:
            # Combine initialization values if present
            init_value = member[2].strip() or member[3].strip() or ""
            init_value = init_value.replace("{", "").replace("}", "")

            comment = member[4].strip() if member[4] else ""
            minmax_match = minmax_pattern.search(comment)

            members.append(Member(
                type     = member[0],
                name     = member[1],
                default_value  = init_value,
                min_max  = minmax_match.group(1) if minmax_match else "",
                optional = bool(optional_pattern.search(comment)),
                disabled = bool(disabled_pattern.search(comment)),
                hidden   = bool(hidden_pattern.search(comment)),
            ))

        adapter_classes.append(generate_adapter_class(struct_name, members))

    header_basename = os.path.basename(file_path)
    lines: list[str] = []
    lines.append("// auto-generated Qt/QML config adapters")
    lines.append("#pragma once")
    lines.append("")
    lines.append("#include <QObject>")
    lines.append("#include <QVariant>")
    lines.append("#include <QString>")
    lines.append("#include <QMetaType>")
    lines.append("")
    # base class, lives in ui plugin config_adapter.h (which is on the include path)
    lines.append("#include \"models/config_adapter.h\"")
    # pull in the original config header so the struct types are visible
    lines.append(f"#include \"{header_basename}\"")
    lines.append("")

    # keep the namespace from the input file if there is one
    ns_match = re.search(r'namespace\s+([A-Za-z_][\w:]*)\s*{', input_text)

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

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python generate-qt-adapters.py <filename1> <filename2> ...")
        sys.exit(1)

    for file_name in sys.argv[1:]:
        with open(file_name, 'r', encoding='utf-8') as input_file:
            file_content = input_file.read()
            generated_content = process_cpp_header(file_content, file_name)
            generated_file_name = file_name.replace('.h', '.gen.h')
            with open(generated_file_name, 'w', encoding='utf-8') as output_file:
                output_file.write(generated_content)
            print("--", generated_file_name)
