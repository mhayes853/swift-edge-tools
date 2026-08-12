# Needle 2 Artifact Bundle

`build-artifact.sh` packages the native Needle 2 static libraries published on Hugging Face.
The model version, immutable repository revision, and checksums form a single dependency lock.
This means a rebuild always uses the exact binaries that were reviewed, even after newer binaries
are published.

To adopt a new release, update `version` and `revision` in `build-artifact.sh`, create the matching
`checksums-<version>.txt`, update `info.json` when its platforms change, and rebuild the bundle.
Checksum failures without those intentional updates indicate that the locked artifacts cannot be
reproduced and should not be accepted automatically.
