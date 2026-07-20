#!/usr/bin/env python3
"""Deploy a built package to a backblaze b2 bucket.

usage: deploy.py <bucket_name> <module> <package_file>

Writes a sha256 checksum file next to the package, then uploads both to
<branch>/<module>/<package-file-name>[.sha256] in the bucket, where <branch>
is the current git branch of this repo.

Credentials are read from the B2_APPLICATION_KEY_ID and B2_APPLICATION_KEY
environment variables.
"""

import base64
import hashlib
import json
import os
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request

AUTH_URL = "https://api.backblazeb2.com/b2api/v2/b2_authorize_account"
UPLOAD_ATTEMPTS = 3
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def api_call(url, auth_token, body=None):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data=data, headers={"Authorization": auth_token})
    try:
        with urllib.request.urlopen(request) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        sys.exit(f"b2 api error from {url}: {error} {error.read().decode(errors='replace')}")


def upload_file(api_url, account_token, bucket_id, local_file, remote_name):
    with open(local_file, "rb") as f:
        content = f.read()
    content_sha1 = hashlib.sha1(content).hexdigest()

    # b2 upload urls can go stale or return transient 5xx,
    # the documented recovery is to fetch a fresh url and retry
    for attempt in range(1, UPLOAD_ATTEMPTS + 1):
        upload = api_call(
            f"{api_url}/b2api/v2/b2_get_upload_url", account_token, {"bucketId": bucket_id}
        )
        request = urllib.request.Request(
            upload["uploadUrl"],
            data=content,
            headers={
                "Authorization": upload["authorizationToken"],
                "X-Bz-File-Name": urllib.parse.quote(remote_name, safe="/"),
                "Content-Type": "b2/x-auto",
                "X-Bz-Content-Sha1": content_sha1,
            },
        )
        try:
            with urllib.request.urlopen(request):
                pass
            return
        except (urllib.error.URLError, OSError) as error:
            print(f"upload attempt {attempt}/{UPLOAD_ATTEMPTS} failed: {error}")
    sys.exit(f"upload failed: {remote_name}")


def main():
    if len(sys.argv) != 4:
        sys.exit(f"usage: {os.path.basename(sys.argv[0])} <bucket_name> <module> <package_file>")
    bucket_name, module, package_file = sys.argv[1:]

    if not os.path.isfile(package_file):
        sys.exit(f"package not found: {package_file} (build the `package` target first)")

    key_id = os.environ.get("B2_APPLICATION_KEY_ID")
    key = os.environ.get("B2_APPLICATION_KEY")
    if not key_id or not key:
        sys.exit("B2_APPLICATION_KEY_ID and B2_APPLICATION_KEY must be set")

    branch = subprocess.run(
        ["git", "-C", REPO_ROOT, "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    remote_prefix = f"{branch}/{module}"
    package_name = os.path.basename(package_file)

    # checksum in `sha256sum -c` format, written next to the package
    with open(package_file, "rb") as f:
        package_sha256 = hashlib.sha256(f.read()).hexdigest()
    sha256_file = f"{package_file}.sha256"
    with open(sha256_file, "w") as f:
        f.write(f"{package_sha256}  {package_name}\n")

    basic_auth = base64.b64encode(f"{key_id}:{key}".encode()).decode()
    auth = api_call(AUTH_URL, f"Basic {basic_auth}")

    buckets = api_call(
        f"{auth['apiUrl']}/b2api/v2/b2_list_buckets",
        auth["authorizationToken"],
        {"accountId": auth["accountId"], "bucketName": bucket_name},
    )["buckets"]
    if not buckets:
        sys.exit(f"bucket not found (or key has no access to it): {bucket_name}")
    bucket_id = buckets[0]["bucketId"]

    for local_file in (sha256_file, package_file):
        remote_name = f"{remote_prefix}/{os.path.basename(local_file)}"
        print(f"uploading b2://{bucket_name}/{remote_name}")
        upload_file(auth["apiUrl"], auth["authorizationToken"], bucket_id, local_file, remote_name)

    print(f"deployed {package_name} to b2://{bucket_name}/{remote_prefix}/")


if __name__ == "__main__":
    main()
