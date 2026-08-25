# Needle 2 Artifact Bundle

`build-artifact.sh` packages the native Needle 2 static libraries published on Hugging Face.
The model version, immutable repository revision, and checksums form a single dependency lock.
This means a rebuild always uses the exact binaries that were reviewed, even after newer binaries
are published.

To adopt a new release, update `version`, `revision`, the target table in `build-artifact.sh`, and
the matching `checksums-<version>.txt`, then rebuild the bundle.
Checksum failures without those intentional updates indicate that the locked artifacts cannot be
reproduced and should not be accepted automatically.

The build also creates host-family ZIPs and an `.artifactbundleindex` next to the monolithic ZIP.
These files are release assets and are intentionally ignored by Git.
