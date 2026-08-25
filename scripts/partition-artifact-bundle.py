#!/usr/bin/env python3

import argparse
import copy
import hashlib
import json
import shutil
import stat
import zipfile
from pathlib import Path


ARCHIVE_TIMESTAMP = (2020, 1, 1, 0, 0, 0)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Partition a static-library artifact bundle into host-family release assets."
    )
    parser.add_argument(
        "bundle", type=Path, help="The expanded .artifactbundle directory."
    )
    parser.add_argument(
        "archive",
        type=Path,
        help="The monolithic .artifactbundle.zip path used to derive output names.",
    )
    return parser.parse_args()


def platform_family(triple):
    if "-apple-" in triple:
        return "apple"
    if "-linux-" in triple:
        return "linux"
    if "-windows-" in triple:
        return "windows"
    raise ValueError(f"unsupported artifact triple: {triple}")


def is_host_triple(family, triple):
    # Index selection uses the build host, so cross-compiled variants ride with their host family.
    if family == "apple":
        return "-apple-macosx" in triple
    if family == "linux":
        return "-unknown-linux-gnu" in triple
    if family == "windows":
        return "-unknown-windows-msvc" in triple
    return False


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archive_info(path, mode):
    info = zipfile.ZipInfo(path, ARCHIVE_TIMESTAMP)
    info.create_system = 3
    info.external_attr = (mode & 0xFFFF) << 16
    return info


def write_directory(archive, path):
    archive.writestr(archive_info(path.rstrip("/") + "/", stat.S_IFDIR | 0o755), b"")


def write_file(archive, source, destination):
    mode = stat.S_IMODE(source.stat().st_mode)
    info = archive_info(destination, stat.S_IFREG | mode)
    info.compress_type = zipfile.ZIP_DEFLATED
    with (
        source.open("rb") as input_file,
        archive.open(info, "w", force_zip64=True) as output_file,
    ):
        shutil.copyfileobj(input_file, output_file, length=1024 * 1024)


def write_bundle_archive(bundle, metadata, included_roots, destination):
    with zipfile.ZipFile(
        destination, "w", zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        write_directory(archive, bundle.name)
        info_path = f"{bundle.name}/info.json"
        info_data = (json.dumps(metadata, indent=2) + "\n").encode()
        info = archive_info(info_path, stat.S_IFREG | 0o644)
        info.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(
            info, info_data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9
        )

        for path in sorted(bundle.rglob("*")):
            relative = path.relative_to(bundle)
            if relative == Path("info.json") or relative.parts[0] not in included_roots:
                continue
            destination_path = f"{bundle.name}/{relative.as_posix()}"
            if path.is_dir():
                write_directory(archive, destination_path)
            elif path.is_file():
                write_file(archive, path, destination_path)
            else:
                raise ValueError(f"unsupported bundle entry: {path}")


def validate_bundle_archive(bundle, metadata, archive_path):
    with zipfile.ZipFile(archive_path) as archive:
        invalid_entry = archive.testzip()
        if invalid_entry:
            raise ValueError(f"corrupt archive entry: {invalid_entry}")

        names = set(archive.namelist())
        prefix = f"{bundle.name}/"
        if any(not name.startswith(prefix) for name in names):
            raise ValueError(f"{archive_path} contains an entry outside {bundle.name}")

        archived_metadata = json.loads(archive.read(f"{prefix}info.json"))
        if archived_metadata != metadata:
            raise ValueError(f"{archive_path} contains unexpected artifact metadata")

        for artifact in metadata["artifacts"].values():
            for variant in artifact["variants"]:
                required_paths = [variant["path"]]
                static_metadata = variant.get("staticLibraryMetadata", {})
                required_paths.extend(static_metadata.get("headerPaths", []))
                module_map_path = static_metadata.get("moduleMapPath")
                if module_map_path:
                    required_paths.append(module_map_path)
                for required_path in required_paths:
                    required_archive_path = f"{prefix}{required_path}"
                    directory_path = f"{required_archive_path.rstrip('/')}/"
                    if (
                        required_archive_path not in names
                        and directory_path not in names
                    ):
                        raise ValueError(
                            f"{required_archive_path} is missing from {archive.filename}"
                        )


def partition_metadata(metadata):
    families = {}
    variant_roots = {}

    for artifact_name, artifact in metadata["artifacts"].items():
        if artifact["type"] != "staticLibrary":
            raise ValueError(f"{artifact_name} is not a static-library artifact")
        for variant in artifact["variants"]:
            triples = variant.get("supportedTriples", [])
            if not triples:
                raise ValueError(
                    f"{artifact_name} has a variant without supported triples"
                )
            variant_families = {platform_family(triple) for triple in triples}
            if len(variant_families) != 1:
                raise ValueError(
                    f"{artifact_name} has a variant spanning platform families"
                )
            family = variant_families.pop()
            root = Path(variant["path"]).parts[0]
            previous_family = variant_roots.setdefault(root, family)
            if previous_family != family:
                raise ValueError(f"variant root {root} spans platform families")
            families.setdefault(family, {}).setdefault(artifact_name, []).append(
                variant
            )

    return families, variant_roots


def sliced_metadata(metadata, variants):
    result = copy.deepcopy(metadata)
    result["artifacts"] = {}
    for artifact_name, artifact_variants in variants.items():
        artifact = copy.deepcopy(metadata["artifacts"][artifact_name])
        artifact["variants"] = artifact_variants
        result["artifacts"][artifact_name] = artifact
    return result


def main():
    arguments = parse_arguments()
    bundle = arguments.bundle.resolve()
    archive = arguments.archive.resolve()
    suffix = ".artifactbundle.zip"

    if not bundle.is_dir() or bundle.suffix != ".artifactbundle":
        raise ValueError(
            f"bundle must be an expanded .artifactbundle directory: {bundle}"
        )
    if not archive.name.endswith(suffix):
        raise ValueError(f"archive must end in {suffix}: {archive}")

    metadata = json.loads((bundle / "info.json").read_text())
    if metadata.get("schemaVersion") != "1.0":
        raise ValueError("only artifact bundle schema version 1.0 is supported")

    families, variant_roots = partition_metadata(metadata)
    shared_roots = {
        path.name
        for path in bundle.iterdir()
        if path.name != "info.json" and path.name not in variant_roots
    }
    output_stem = archive.name[: -len(suffix)]
    index_path = archive.with_name(f"{output_stem}.artifactbundleindex")
    archives = []

    for family in sorted(families):
        variants = families[family]
        host_triples = sorted(
            {
                triple
                for artifact_variants in variants.values()
                for variant in artifact_variants
                for triple in variant["supportedTriples"]
                if is_host_triple(family, triple)
            }
        )
        if not host_triples:
            raise ValueError(f"the {family} slice has no supported build-host triple")

        family_roots = {
            Path(variant["path"]).parts[0]
            for artifact_variants in variants.values()
            for variant in artifact_variants
        }
        archive_path = archive.with_name(f"{output_stem}-{family}{suffix}")
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        family_metadata = sliced_metadata(metadata, variants)
        write_bundle_archive(
            bundle,
            family_metadata,
            shared_roots | family_roots,
            archive_path,
        )
        validate_bundle_archive(bundle, family_metadata, archive_path)
        archives.append(
            {
                "fileName": archive_path.name,
                "checksum": sha256(archive_path),
                "supportedTriples": host_triples,
            }
        )
        print(
            f"Built {archive_path} ({archive_path.stat().st_size / 1024 / 1024:.1f} MiB)."
        )

    index = {"schemaVersion": "1.0", "archives": archives}
    index_path.write_text(json.dumps(index, indent=2) + "\n")
    print(f"Built {index_path}.")
    print(f"Index checksum: {sha256(index_path)}")


if __name__ == "__main__":
    main()
