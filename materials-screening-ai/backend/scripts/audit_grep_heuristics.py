# -*- coding: utf-8 -*-
"""
Auditing script to grep for heuristic literals, fake hull/gap constants,
and old hardcoded metrics across the entire codebase.
"""
import os
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

TERMS = [
    '-5.50',
    '0.85',
    'est_gap',
    'is_solar_optimal',
    'e_above_hull',
    '0.532215',
    '81.0',
    '88.0',
    '27.3',
]

SEARCH_DIRS = [
    'materials-screening-ai/backend',
    'materials-screening-ai/frontend',
    'materials-screening-ai/research'
]

def main():
    print("================================================================================")
    print("CODEBASE GREP AUDIT FOR HEURISTICS, LITERALS, AND HARDCODED METRICS")
    print("================================================================================")
    for term in TERMS:
        print(f"\n>>> SEARCH RESULTS FOR: '{term}'")
        matches = []
        for sdir in SEARCH_DIRS:
            for root, dirs, files in os.walk(sdir):
                if any(x in root for x in ['.git', 'node_modules', '.next', '__pycache__']):
                    continue
                for f in files:
                    if f.endswith(('.py', '.ts', '.tsx', '.json', '.md', '.csv', '.ini')):
                        fpath = os.path.join(root, f)
                        try:
                            with open(fpath, 'r', encoding='utf-8', errors='ignore') as fl:
                                for lno, line in enumerate(fl, 1):
                                    if term in line:
                                        matches.append((fpath, lno, line.strip()))
                        except Exception:
                            pass
        if matches:
            for fpath, lno, line in matches:
                print(f"  {fpath}:{lno}: {line[:120]}")
            print(f"  [Total occurrences of '{term}': {len(matches)}]")
        else:
            print(f"  [ZERO OCCURRENCES FOUND FOR '{term}']")

if __name__ == '__main__':
    main()
