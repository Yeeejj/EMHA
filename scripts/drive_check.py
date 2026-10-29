"""
Confirm v3 CNN-path draws on the Google Drive server (Colab only).

In three Colab sessions the mounted Drive accepted perm_<k>.csv files, the
runner counted them, and they never reached the Drive server. The draws
notebook therefore stages the draws on the local disk, flushes Drive, and
calls check_draws(): it lists the draw folder through the Drive API, uploads
every staged draw the server does not have, and lists again. A draw counts as
done only when the server holds exactly one file of that name with the staged
size.

The functions take a Drive v3 service object (drive_service() in Colab, after
google.colab.auth.authenticate_user()); googleapiclient is imported only
inside them, so this module imports without it. Folders are addressed by
their path relative to My Drive.
"""

from __future__ import annotations

import time
from pathlib import Path

FOLDER_MIME = "application/vnd.google-apps.folder"
DRAW_GLOB = "perm_*.csv"
_PAGE_SIZE = 1000
_CONFIRM_WAIT_S = 300
_CONFIRM_POLL_S = 15


def drive_service():
    """Drive v3 client with the Colab user's credentials."""
    from googleapiclient.discovery import build

    return build("drive", "v3", cache_discovery=False)


def drive_folder_id(svc, rel: str) -> str:
    """Id of the folder at `rel` (relative to My Drive); exactly one per level."""
    parent = "root"
    for part in rel.split("/"):
        q = (
            f"'{parent}' in parents and name = '{part}' and trashed = false "
            f"and mimeType = '{FOLDER_MIME}'"
        )
        found = svc.files().list(q=q, fields="files(id)").execute()["files"]
        if len(found) != 1:
            raise RuntimeError(
                f"Drive folder {rel}: {len(found)} folders named {part!r}"
            )
        parent = found[0]["id"]
    return parent


def drive_sizes(svc, folder: str) -> dict:
    """{name: [size, ...]} of the files in `folder` on the Drive server."""
    sizes, token = {}, None
    while True:
        res = (
            svc.files()
            .list(
                q=f"'{folder}' in parents and trashed = false",
                fields="nextPageToken, files(name, size)",
                pageSize=_PAGE_SIZE,
                pageToken=token,
            )
            .execute()
        )
        for f in res["files"]:
            sizes.setdefault(f["name"], []).append(int(f.get("size", 0)))
        token = res.get("nextPageToken")
        if not token:
            return sizes


def drive_upload(svc, folder: str, path, mimetype: str) -> None:
    """Upload the local file `path` into `folder` under its own name."""
    from googleapiclient.http import MediaFileUpload

    media = MediaFileUpload(str(path), mimetype=mimetype, resumable=True)
    body = {"name": Path(path).name, "parents": [folder]}
    svc.files().create(body=body, media_body=media, fields="id").execute()


def confirm_draw(
    svc,
    folder: str,
    path,
    wait_s: int = _CONFIRM_WAIT_S,
    poll_s: int = _CONFIRM_POLL_S,
    sleep=time.sleep,
) -> tuple[str, bool]:
    """Confirm one draw on the server right after it is written; (line, ok).

    Waits up to wait_s for the mounted Drive to upload `path`. If no file of
    that name is on the server by then, uploads it through the API. ok means
    the server holds exactly one file of that name with the local size; a
    file of that name with another size or a duplicate is never overwritten.
    """
    path = Path(path)
    size = path.stat().st_size
    waited = 0
    while True:
        have = drive_sizes(svc, folder).get(path.name, [])
        if have == [size]:
            return f"drive server: {path.name} confirmed ({size} bytes)", True
        if waited >= wait_s:
            break
        sleep(poll_s)
        waited += poll_s
    if have:
        return f"drive server: {path.name} MISMATCH: server {have}, local {size}", False
    drive_upload(svc, folder, path, "text/csv")
    have = drive_sizes(svc, folder).get(path.name, [])
    line = f"drive server: {path.name} uploaded through the API ({size} bytes)"
    if have != [size]:
        return f"{line}; server now {have}", False
    return line, True


def check_draws(svc, staging, rel: str, n) -> tuple[list, bool]:
    """Upload staged draws missing on the server; return (report lines, ok).

    ok is False if a staged draw is missing or has another size on the
    server, or if any draw name appears more than once there.
    """
    folder = drive_folder_id(svc, rel)
    staging = Path(staging)
    local = {p.name: p.stat().st_size for p in sorted(staging.glob(DRAW_GLOB))}
    lines = [f"drive server: checking {len(local)} draws from this runtime"]
    before = drive_sizes(svc, folder)
    for name, size in local.items():
        if name not in before:
            drive_upload(svc, folder, staging / name, "text/csv")
            lines.append(f"drive server: uploaded {name} ({size} bytes)")
    server = drive_sizes(svc, folder)
    bad = [name for name, size in local.items() if server.get(name) != [size]]
    perms = sorted(k for k in server if Path(k).match(DRAW_GLOB))
    bad += [k for k in perms if len(server[k]) > 1 and k not in bad]
    lines.append(f"drive server: {len(perms)}/{n} present in {rel}")
    if bad:
        lines.append(
            f"drive server: MISMATCH (missing, wrong size or duplicate): {bad}"
        )
    return lines, not bad
