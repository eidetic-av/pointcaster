import os
import re
import sys

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

def calculate_relative_path(from_file, to_file):
    from_dir = os.path.dirname(from_file)
    relative_path = os.path.relpath(to_file, from_dir)
    return relative_path

def ensure_headers(file_content, header_paths):
    lines = file_content.splitlines()

    # find the index of the last #include
    last_include_index = next((i for i, line in reversed(list(enumerate(lines))) if line.startswith('#include')), -1)

    # format the header paths into includes
    headers_to_insert = [f'#include "{header}"' for header in header_paths if f'#include "{header}"' not in file_content]

    # insert the headers after the last #include
    if headers_to_insert:
        insert_index = last_include_index + 1 if last_include_index != -1 else 0
        lines[insert_index:insert_index] = headers_to_insert

    return '\n'.join(lines)

def process_cpp_header(input_text, file_path):
    # structs = re.findall(r"struct (\w+Configuration) {([\s\S]*?^\};)",

    # TODO: replace this explicit string search with a search for a meta comment
    # which instructs to serialize the struct
    structs = re.findall(r"struct (\w+(?:Configuration|Session|SessionLayout|BindingTarget|OutputRoute|ABB)) {([\s\S]*?^\};)",
    # structs = re.findall(r"struct (\w+Configuration) {([\s\S]*?^\};)",
                         input_text, re.MULTILINE)

    modified_text = input_text

    # define regex components
    type_pattern = r"([\w:]+(?:\:\:)?[\w:]+(?:<[\w\s:,]+>)?)"
    name_pattern = r"(\w+)"
    initial_value_pattern = r"(?:\s*=\s*([^;{]+)|\s*{\s*([^}]+)\s*})?"
    comment_pattern = r"(?:\s*//\s*(.*))?"
    minmax_pattern = re.compile(r"@minmax\(([^)]+)\)")
    optional_pattern = re.compile(r"@optional")

    # combine regex into a full matcher for struct members
    member_pattern = re.compile(
        fr"{type_pattern}\s+{name_pattern}{initial_value_pattern}\s*;{comment_pattern}",
        re.VERBOSE)    

    for struct_name, struct_body in structs:

        # find member variables
        unparsed_members = member_pattern.findall(struct_body)
        members = []
        for member in unparsed_members:
            # Combine initialization values if present
            init_value = member[2].strip() or member[3].strip() or ""
            init_value = init_value.replace("{", "").replace("}", "")

            comment = member[4].strip() if member[4] else "" 

            min_max = ""
            minmax_match = minmax_pattern.search(comment)
            if minmax_match:
                min_max = minmax_match.group(1)

            optional = False
            optional_match = optional_pattern.search(comment)
            if optional_match:
                optional = True

            members.append((member[0], member[1],
                            f"{{{init_value}}}", min_max, optional))

         # ensure there is an 'unfolded' bool member for the gui system
        if "Configuration" in struct_name:
            if not any(member[1] == "unfolded" for member in members):
                members.insert(0, ("bool", "unfolded", "{false}", "", False))

        generated_struct = f"struct {struct_name} {{\n"

        def add_member_to_body(member_type, member_name, default_value):
            nonlocal generated_struct
            generated_struct += f"\t{member_type} {member_name}{default_value};\n"

        for member in members:
            add_member_to_body(member[0], member[1], member[2])

        # generate the reflection metadata to add to the struct

        # first the formatted name to be displayed in gui
        format_name = format_struct_name(struct_name)
        generated_struct += f"\n\tstatic constexpr const char* Name = \"{format_name}\";\n"

        # then the count of members
        generated_struct += f"\tstatic constexpr std::size_t MemberCount = {len(members)};\n"

        # the type of each member
        generated_struct += "\tusing MemberTypes = pc::reflect::type_list<"
        all_types_string = ""
        for i, member in enumerate(members):
            generated_struct += member[0]
            all_types_string += member[0]
            if i != len(members) - 1:
                generated_struct += ", "
                all_types_string += ", "
        generated_struct += ">;\n"

        # the default values of each member
        generated_struct += f"\tinline static const std::tuple<{all_types_string}> Defaults {{ "
        for i, member in enumerate(members):
            default_value = member[2]
            generated_struct += f"{default_value}"
            if i != len(members) - 1:
                generated_struct += ", "
        generated_struct += "};\n"

        # the min and max values of each member
        generated_struct += f"\tstatic constexpr std::array<std::optional<pc::types::MinMax<float>>, {len(members)}> MinMaxValues {{\n"
        for i, member in enumerate(members):
            min_max = member[3]
            if min_max == "":
                generated_struct += f"\t\tstd::optional<pc::types::MinMax<float>>{{}}"
            else:
                generated_struct += f"\t\tstd::optional<pc::types::MinMax<float>>{{{{{min_max}}}}}"
            if i != len(members) - 1:
                generated_struct += ", \n"
        generated_struct += "};\n"

        # add an "empty()" member function that returns true if the configuration struct
        # contains no changed values (i.e. each member of the struct is currently equal to its default value)
        if len(members) > 1:
            generated_struct += "\n\tbool empty() const { return \n"
            for i, member in enumerate(members):
                name = member[1]
                if name == "unfolded":
                    continue
                generated_struct += f"\t\t{name} == std::get<{i}>(Defaults)"
                if i != len(members) - 1:
                    generated_struct += " && \n"
                else:
                    generated_struct += "; }\n"

        # add a default comparison operator
        generated_struct += f"\n\tbool operator==(const {struct_name}&) const = default;\n"

        # add the serdepp macro
        generated_struct += f"\n\tDERIVE_SERDE({struct_name},"
        for i, member in enumerate(members):
            member_name = member[1]
            serde_attributes = ""
            is_optional = member[4]
            if (is_optional):
                serde_attributes += ", make_optional"
            generated_struct += f"\n\t\t(&Self::{member_name}, \"{member_name}\"{serde_attributes})"
        generated_struct += ")\n"

        generated_struct += "};\n"

        full_struct = f"struct {struct_name} {{{struct_body}"

        modified_text = modified_text.replace(full_struct, generated_struct)

    # add the required headers to the top of the file if necessary
    required_headers = ['src/serialization.h', 'src/structs.h']
    relative_header_paths = [calculate_relative_path(file_path, header) for header in required_headers]

    # check / add the headers in this file
    modified_text = ensure_headers(modified_text, relative_header_paths)

    return(modified_text)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python generate_reflections.py <filename1> <filename2> ...")
        sys.exit(1)

    for file_name in sys.argv[1:]:
        with open(file_name, 'r') as input_file:
            file_content = input_file.read()
            modified_content = process_cpp_header(file_content, file_name)
            generated_file_name = file_name.replace('.h', '.gen.h')
            with open(generated_file_name, 'w') as output_file:
                output_file.write(modified_content)
            print("--", generated_file_name)
