This is a minimal compatibility shim for the legacy `bitflags` 0.8 macro syntax used by
`ladspa` 0.3.4. It is patched workspace-wide because the original crate no longer compiles with
the current dependency graph.

The shim intentionally implements only the API exercised by `ladspa`. It is covered by the
LADSPA default and all-feature CI builds and should be removed when the LADSPA dependency is
updated or replaced.

License: MIT OR Apache-2.0, matching the package metadata and repository licenses.
