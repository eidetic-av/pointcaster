#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Merge vcpkg licence files into a single LICENSE output."
    )
    parser.add_argument(
        '--target',
        required=True,
        help="CMake binary directory (build directory), e.g. build/windows-release"
    )
    parser.add_argument(
        '--triplet',
        required=True,
        help="vcpkg target triplet, e.g. x64-windows-static-md-release"
    )
    parser.add_argument(
        '--output',
        required=True,
        help="Output directory for the generated LICENSE file (target file directory)"
    )
    return parser.parse_args()

def main():
    args = parse_arguments()
    target_dir = Path(args.target)
    triplet = args.triplet
    output_dir = Path(args.output)

    share_dir = target_dir / "vcpkg_installed" / triplet / "share"
    if not share_dir.is_dir():
        print(f"Error: Share directory '{share_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    # Define licence filenames (case-insensitive)
    licence_names = {"license", "license.md",
                     "copyright", "copyright.md",
                     "notices", "notices.md", "notice", "notice.md"}

    output_file = output_dir / "NOTICE.txt"
    output_lines = []

    # Iterate over each library folder in the share directory
    for library_dir in sorted(share_dir.iterdir()):
        if not library_dir.is_dir():
            continue
        licence_found = False
        for file in sorted(library_dir.iterdir()):
            if file.is_file() and file.name.lower() in licence_names:
                if not licence_found:
                    header = f"\n----- Licence from '{library_dir.name}' -----\n\n"
                    output_lines.append(header)
                    licence_found = True
                try:
                    content = file.read_text(encoding="utf-8")
                except Exception as e:
                    print(f"Warning: Could not read {file}: {e}", file=sys.stderr)
                    continue
                output_lines.append(content)
                output_lines.append("\n")

    new_content = "".join(output_lines)

    # Only replace the file if content is different
    if output_file.exists():
        try:
            old_content = output_file.read_text(encoding="utf-8")
        except Exception as e:
            print(f"Warning: Could not read existing file {output_file}: {e}", file=sys.stderr)
            old_content = None
        if old_content == new_content:
            # print(f"No changes to licenses, not updating {output_file}.")
            return

    try:
        with output_file.open("w", encoding="utf-8") as out:
            out.write(new_content)
    except Exception as e:
        print(f"Error: Could not write to {output_file}: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Generated licence file at: {output_file}")

if __name__ == '__main__':
    main()
