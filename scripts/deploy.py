#!/usr/bin/env python3
import argparse
import os
import zipfile
import hashlib
import subprocess
import fnmatch
import shutil
import sys

def zip_directory(directory, zip_path, exclude_patterns):
    """Zip all files in the directory into zip_path without a top-level directory.
    
    Files matching any of the exclude_patterns are skipped.
    """
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(directory):
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, directory)
                normalized_path = rel_path.replace(os.sep, '/')
                if any(fnmatch.fnmatch(normalized_path, pattern) for pattern in exclude_patterns):
                    continue
                zipf.write(full_path, rel_path)

def compute_sha256(file_path):
    """Compute and return the SHA256 hash of the file."""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()

def upload_file(bucket, local_file, remote_file):
    """Invoke B2 CLI to upload a file to the specified bucket."""
    cmd = ['b2', 'file', 'upload', bucket, local_file, remote_file]
    subprocess.run(cmd, check=True)

def main():
    # Check required executables before proceeding.
    required = ["git", "b2"]
    for exe in required:
        if shutil.which(exe) is None:
            print(f"Error: Required executable '{exe}' not found in PATH.")
            sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Zip directory contents, generate a checksum, and upload to Backblaze B2."
    )
    parser.add_argument('--name', required=True,
                        help='Deployment name (include target/commit info if needed)')
    parser.add_argument('--directory', required=True,
                        help='Directory whose contents will be zipped')
    parser.add_argument('--exclude', nargs='*', default=[],
                        help='List of glob patterns to exclude from the archive')
    parser.add_argument('--bucket', required=True,
                        help='Backblaze B2 bucket name')
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"Error: {args.directory} is not a valid directory.")
        sys.exit(1)

    # the git commit short hash gets appended to the artefact name
    try:
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(args.directory)
        ).strip().decode()
    except subprocess.CalledProcessError:
        print("Error: Unable to get git commit hash. Ensure you're running this within a git repository.")
        sys.exit(1)

    zip_filename = f"/tmp/{args.name}_{commit_hash}.zip"
    sha256_filename = f"{zip_filename}.sha256"

    print(f"Zipping contents of '{args.directory}' into '{zip_filename}'")
    if args.exclude:
        print(f"Excluding patterns: {args.exclude}")
    zip_directory(args.directory, zip_filename, args.exclude)
    
    print("Computing SHA256 checksum")
    hash_value = compute_sha256(zip_filename)
    with open(sha256_filename, 'w') as f:
        f.write(f"{hash_value}  {os.path.basename(zip_filename)}\n")

    print(f"Uploading '{zip_filename}' to bucket '{args.bucket}'")
    upload_file(args.bucket, zip_filename, os.path.basename(zip_filename))

    print(f"Uploading '{sha256_filename}' to bucket '{args.bucket}'")
    upload_file(args.bucket, sha256_filename, os.path.basename(sha256_filename))

    try:
        if os.path.exists(zip_filename):
            os.remove(zip_filename)
        if os.path.exists(sha256_filename):
            os.remove(sha256_filename)
        print("Temporary files deleted.")
    except Exception as e:
        print(f"Warning: Could not remove temporary file(s): {e}")

    print("Deployment complete.")

if __name__ == "__main__":
    main()
