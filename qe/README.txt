Quantum ESPRESSO for Windows (Intel oneAPI + MS-MPI) — Precompiled Binaries

Build date (UTC): 2025-12-23 08:40:07 UTC
Quantum ESPRESSO version/tag: qe-7.5

Why this exists (and why it matters)

Building Quantum ESPRESSO on Windows is significantly harder than on Linux/macOS:
toolchains, Fortran runtime DLLs, MPI, and dependency closure can be painful even for experts.

This distribution is a "just run it" package:
- All executables and required runtime DLLs are colocated in bin/
- No extra compiler/runtime installation needed on a typical Windows machine
- Signed executables (when built via CI release workflow) for safer downloading and sharing

Contents

Contains 109 executable(s) in bin/ and 13 runtime DLL(s) in bin/.

Directory layout

- bin/
  Quantum ESPRESSO executables (*.exe), bundled runtime DLL dependencies, and (optionally) MS-MPI launcher.

- licenses/
  License texts for Quantum ESPRESSO and bundled third-party components.
  See licenses/THIRD_PARTY_NOTICES.txt for a summary.

- VERSION.txt
  Build metadata for traceability (exact build flags, toolchain versions, etc.)

Quick start (serial)

1) Open PowerShell or CMD.
2) cd into the bin\ folder:
     cd .\bin
3) Run an executable, e.g.:
     .\pw.exe -i scf-cg.in > scf-cg.out

Optional: add bin/ to PATH for this terminal session only:

  set "PATH=%CD%;%PATH%"               (CMD, when you are already inside bin\)
  $env:PATH = "$PWD;$env:PATH"         (PowerShell, when you are already inside bin\)

If you are NOT inside bin\ yet, use:

  set "PATH=%CD%\bin;%PATH%"           (CMD)
  $env:PATH = "$PWD\bin;$env:PATH"     (PowerShell)

MPI usage

MS-MPI launcher is included (mpiexec/msmpiexec) in bin/.

Tip:
- If you run from inside bin\, mpiexec should be found automatically.
- Otherwise, add bin/ to PATH for the current terminal session (see the PATH hint above).

Example (MPI): run from inside bin\:

  mpiexec -n 4 .\pw.exe -i scf-cg.in > scf-cg.out

OpenMP / threading

Set OMP_NUM_THREADS to control OpenMP threading:

  set OMP_NUM_THREADS=8                (CMD)
  $env:OMP_NUM_THREADS=8               (PowerShell)

Set MKL_NUM_THREADS to control Intel MKL threading (recommended when using OpenMP):

  set MKL_NUM_THREADS=8                (CMD)
  $env:MKL_NUM_THREADS=8               (PowerShell)

Tip: For hybrid MPI+OpenMP runs, avoid oversubscription.
Example: on a 16-core CPU, try mpiexec -n 4 with OMP_NUM_THREADS=4 as a starting point. Tune the above parameters to find the best performance for your calculations.

About QMatSuite (GUI)

If you want a more user-friendly workflow (modern user interface, project management, input generation, job management, result browsing),
check out QMatSuite — a modern GUI that can drive Quantum ESPRESSO and other engines. It is free and open source.

- Project home / downloads: www.qmatsuite.com
- Source code / releases: github.com/QMatSuite

This repo/toolchain not only produces ready-to-run executables, but also serves as the "high-performance, reproducible build backend" for QMatSuite,
and as a reference compilation recipe for anyone exploring QE/Wannier/etc. on Windows.

Licensing (important)

- Quantum ESPRESSO is licensed under GPL v2 or later.
- Bundled third-party runtime components (e.g., Intel oneAPI runtimes, Microsoft MPI, MSVC runtime DLLs)
  are redistributed under their respective licenses.
  See the licenses/ folder for the exact texts.

Disclaimer

These binaries are provided "as is" without warranty.

If you report an issue, please include:
- Windows version
- CPU model
- Whether you used MPI and/or OpenMP (and the values of OMP_NUM_THREADS / MKL_NUM_THREADS)
- The command line used (including mpiexec arguments)
- The contents of VERSION.txt