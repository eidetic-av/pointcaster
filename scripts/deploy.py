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

def b2_upload_file(bucket, local_file, remote_file):
    """Invoke B2 CLI to upload a file to the specified bucket."""
    cmd = ['b2', 'file', 'upload', bucket, local_file, remote_file]
    subprocess.run(cmd, check=True)

def upload_directory(bucket, directory, exclude_patterns):
    """Recursively upload files in the directory to the specified B2 bucket.
    
    Each file is uploaded with its path relative to the directory root.
    """
    for root, _, files in os.walk(directory):
        for file in files:
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, directory).replace(os.sep, '/')
            if any(fnmatch.fnmatch(rel_path, pattern) for pattern in exclude_patterns):
                continue
            print(f"Uploading {full_path} to bucket {bucket} as {rel_path}")
            b2_upload_file(bucket, full_path, rel_path)

def main():
    parser = argparse.ArgumentParser(
        description="Deploy assets by optionally archiving them and transferring via Backblaze B2 or SCP."
    )
    parser.add_argument('--directory', required=True,
                        help='Directory whose contents will be processed')
    parser.add_argument('--exclude', nargs='*', default=[],
                        help='List of glob patterns to exclude from processing')
    parser.add_argument('--archive', action='store_true',
                        help='If set, zip the directory contents and transfer the archive')
    parser.add_argument('--name',
                        help='If set (and with --archive), set the archive output name')
    parser.add_argument('--sha256', action='store_true',
                        help='If set (and with --archive), compute and upload a SHA256 checksum file')
    parser.add_argument('destinations', nargs='+',
                        help='List of deploy destinations (b2://bucket for B2 upload or remote address for SCP)')
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"Error: {args.directory} is not a valid directory.")
        sys.exit(1)

    b2_destinations = [d for d in args.destinations if d.startswith("b2://")]
    remote_destinations = [d for d in args.destinations if not d.startswith("b2://")]

    # check we have all the external dependencies we need
    if shutil.which("git") is None:
        print(f"Error: Required executable 'git' not found in PATH.")
        sys.exit(1)
    if b2_destinations:
        if shutil.which("b2") is None:
            print(f"Error: Required executable 'b2' not found in PATH.")
            sys.exit(1)
    if remote_destinations:
        if shutil.which("scp") is None:
            print(f"Error: Required executable 'scp' not found in PATH.")
            sys.exit(1)

    # if --archive is specified, bundle directory contents into a zip archive.
    if args.archive:
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=os.path.dirname(args.directory)
        ).strip().decode()

        zip_filename = f"/tmp/{args.name}_{commit_hash}.zip"
        print(f"Zipping contents of '{args.directory}' into '{zip_filename}'")
        if args.exclude:
            print(f"Excluding patterns: {args.exclude}")
        zip_directory(args.directory, zip_filename, args.exclude)

        if args.sha256:
            print("Computing SHA256 checksum")
            hash_value = compute_sha256(zip_filename)
            sha256_filename = f"{zip_filename}.sha256"
            with open(sha256_filename, 'w') as f:
                f.write(f"{hash_value}  {os.path.basename(zip_filename)}\n")

        # For each B2 destination, upload the archive (and checksum if requested)
        for dest in b2_destinations:
            bucket = dest[len("b2://"):]
            print(f"Uploading '{zip_filename}' to bucket '{bucket}'")
            b2_upload_file(bucket, zip_filename, os.path.basename(zip_filename))
            if args.sha256:
                print(f"Uploading '{sha256_filename}' to bucket '{bucket}'")
                b2_upload_file(bucket, sha256_filename, os.path.basename(sha256_filename))
        # For each remote destination, use scp to copy the archive
        if remote_destinations:
            for dest in remote_destinations:
                print(f"Copying archive '{zip_filename}' to remote destination '{dest}'")
                scp_cmd = ["scp", zip_filename, dest]
                subprocess.run(scp_cmd, check=True)
        try:
            if os.path.exists(zip_filename):
                os.remove(zip_filename)
            if args.sha256 and os.path.exists(sha256_filename):
                os.remove(sha256_filename)
            print("Temporary files deleted.")
        except Exception as e:
            print(f"Warning: Could not remove temporary file(s): {e}")

        print("Deployment complete (archive).")
    else:
        # --archive not specified: transfer raw files.
        if args.sha256:
            print("Warning: --sha256 has no effect when --archive is not specified.")
        # For each B2 destination, recursively upload files.
        for dest in b2_destinations:
            if shutil.which("b2") is None:
                print("Error: Required executable 'b2' not found in PATH.")
                sys.exit(1)
            bucket = dest[len("b2://"):]
            print(f"Uploading raw files from '{args.directory}' to bucket '{bucket}'")
            upload_directory(bucket, args.directory, args.exclude)
        # For each remote destination, use scp to recursively copy the directory.
        if remote_destinations:
            for dest in remote_destinations:
                print(f"Copying raw contents of '{args.directory}' to remote destination '{dest}'")
                # Create a list of files to send, respecting exclusions
                files_to_send = []
                for root, _, files in os.walk(args.directory):
                    for file in files:
                        rel_path = os.path.relpath(os.path.join(root, file), args.directory)
                        normalized_path = rel_path.replace(os.sep, '/')
                        if not any(fnmatch.fnmatch(normalized_path, pattern) for pattern in args.exclude):
                            files_to_send.append(os.path.join(root, file))
                if files_to_send:
                    scp_cmd = ["scp"] + files_to_send + [dest]
                    subprocess.run(scp_cmd, check=True)
                print("Deployment complete (raw files).")

if __name__ == "__main__":
    main()
