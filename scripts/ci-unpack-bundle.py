#!/usr/bin/env python3
"""Extract a previously checksummed bundle archive for protected CI only.

This build-time helper accepts regular files and directories, never links or
special entries. It is not used by the Go library or inference process.
"""

import pathlib
import shutil
import sys
import tarfile


MAX_FILES = 32
MAX_BYTES = 4 << 30


def unpack(archive: pathlib.Path, destination: pathlib.Path) -> None:
    if destination.exists():
        raise ValueError("bundle destination already exists")
    destination.mkdir(mode=0o700)
    seen = set()
    total = 0
    files = 0
    with tarfile.open(archive, "r:gz") as source:
        for member in source:
            if member.isdir() and member.name in (".", "./"):
                continue
            name = pathlib.PurePosixPath(member.name)
            if (
                name.is_absolute()
                or not name.parts
                or any(part in ("", ".", "..") for part in name.parts)
                or str(name) in seen
            ):
                raise ValueError("bundle archive has an invalid or duplicate path")
            seen.add(str(name))
            target = destination.joinpath(*name.parts)
            if member.isdir():
                target.mkdir(mode=0o700, parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ValueError("bundle archive contains a link or special entry")
            files += 1
            total += member.size
            if files > MAX_FILES or total > MAX_BYTES:
                raise ValueError("bundle archive exceeds CI extraction limits")
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            stream = source.extractfile(member)
            if stream is None:
                raise ValueError("bundle archive member cannot be read")
            with stream, target.open("xb") as output:
                shutil.copyfileobj(stream, output, length=1 << 20)
            if target.stat().st_size != member.size:
                raise ValueError("bundle archive member size changed during extraction")
    if files == 0 or not (destination / "manifest.json").is_file():
        raise ValueError("bundle archive has no complete bundle root")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: ci-unpack-bundle.py ARCHIVE DESTINATION")
    try:
        unpack(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]))
    except (OSError, tarfile.TarError, ValueError) as error:
        raise SystemExit(f"unsafe or incomplete CI bundle archive: {error}") from error
