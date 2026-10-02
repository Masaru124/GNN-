# Research Reproduction Commands

Run full tests from repository root:

```powershell
$env:PYTHONPATH=(Join-Path $PWD 'materials-screening-ai\backend')
.venv311\Scripts\python.exe -m pytest materials-screening-ai\backend\tests --durations=10 --junitxml=materials-screening-ai\research\junit_root.xml *> materials-screening-ai\research\pytest_root.log
```

Run full tests from `materials-screening-ai/`:

```powershell
..\.venv311\Scripts\python.exe -m pytest --durations=10 --junitxml=research\junit.xml *> research\pytest.log
```

Run QE tests explicitly:

```powershell
$env:QE_BIN_DIR=(Resolve-Path ..\qe\bin).Path
$head=(& git rev-parse HEAD)
"git rev-parse HEAD: $head" | Set-Content research\pytest_qe.log -Encoding utf8
..\.venv311\Scripts\python.exe -m pytest -m qe backend/tests/test_dft_benchmark_regression.py --durations=10 --junitxml=research\junit_qe.xml *>> research\pytest_qe.log
```

Sanitize test artifacts:

```powershell
$files=@('research\pytest.log','research\junit.xml','research\pytest_qe.log','research\junit_qe.xml')
$workspaceRoot=(Resolve-Path ..).Path + '\'
foreach($file in $files){ $text=Get-Content -Raw $file; $text=$text -replace [regex]::Escape($workspaceRoot),'./'; Set-Content -Path $file -Value $text -Encoding utf8NoBOM }
```
