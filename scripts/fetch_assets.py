#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "assets" / "manifest.json"
ASSET_ROOT = REPO_ROOT / "assets" / "booster"


def main() -> int:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    src = manifest["booster_assets"]
    repo = src["repo"]
    rev = src["rev"]
    files = src["files"]

    with tempfile.TemporaryDirectory(prefix="booster_assets_") as td:
        tmp = Path(td)
        clone_dir = tmp / "booster_assets"
        clone_root = clone_dir.resolve()

        try:
            subprocess.run(["git", "clone", repo, str(clone_dir)], check=True)
            subprocess.run(["git", "checkout", rev], cwd=clone_dir, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to fetch revision {rev} from {repo}: {e}", file=sys.stderr)
            return 1

        ASSET_ROOT.mkdir(parents=True, exist_ok=True)

        copied_paths: set[str] = set()
        xml_paths: list[Path] = []

        for rel in files:
            src_path = clone_dir / rel
            if not src_path.exists():
                print(f"Manifest file missing at revision {rev}: {rel}", file=sys.stderr)
                return 2
            dst_path = ASSET_ROOT / rel
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst_path)
            copied_paths.add(rel)
            if src_path.suffix.lower() == ".xml":
                xml_paths.append(src_path)
            print(f"Copied: {rel}")

        # Copy file-based dependencies referenced from XML assets (for example meshes).
        for xml_path in xml_paths:
            tree = ET.parse(xml_path)
            xml_dir = xml_path.parent
            compiler = tree.getroot().find("./compiler")
            meshdir = compiler.get("meshdir") if compiler is not None else None
            texturedir = compiler.get("texturedir") if compiler is not None else None
            for elem in tree.findall(".//*[@file]"):
                rel_file = elem.get("file")
                if not rel_file:
                    continue
                candidates = [xml_dir / rel_file]
                if meshdir and elem.tag == "mesh":
                    candidates.append(xml_dir / meshdir / rel_file)
                if texturedir and elem.tag in {"texture", "material"}:
                    candidates.append(xml_dir / texturedir / rel_file)

                dep_src = None
                for candidate in candidates:
                    if candidate.exists():
                        dep_src = candidate.resolve()
                        break
                if dep_src is None:
                    dep_src = (xml_dir / rel_file).resolve()
                try:
                    dep_rel = dep_src.relative_to(clone_root).as_posix()
                except ValueError:
                    print(
                        f"Referenced dependency escapes source tree: {rel_file}",
                        file=sys.stderr,
                    )
                    return 3
                if not dep_src.exists():
                    print(
                        f"Missing XML dependency at revision {rev}: {dep_rel}",
                        file=sys.stderr,
                    )
                    return 4
                if dep_rel in copied_paths:
                    continue
                dep_dst = ASSET_ROOT / dep_rel
                dep_dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(dep_src, dep_dst)
                copied_paths.add(dep_rel)
                print(f"Copied dependency: {dep_rel}")

    source = {
        "repo": repo,
        "rev": rev,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (ASSET_ROOT / ".source.json").write_text(
        json.dumps(source, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote provenance: {ASSET_ROOT / '.source.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
