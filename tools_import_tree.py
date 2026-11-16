"""
tools_import_tree.py -- what does `python main.py` actually reach?
==================================================================

The repository is meant to contain the code the paper needs and nothing else.
"Nothing else" is a claim that decays unless something checks it, so this
walks the import graph from `main.py` and prints two lists: reachable, and
not. Anything in the second list is either dead code or is reached by a route
this walker cannot see (a subprocess, a plugin lookup) -- and in that case it
should be named here, not left to be rediscovered.

    python tools_import_tree.py
"""

from __future__ import annotations

import ast
import os
import sys

# Reached by running a file rather than by importing it. Stated explicitly so
# "not imported" never silently means "not used".
RUN_NOT_IMPORTED = {
    'tests': 'run as subprocesses by main._run_self_tests',
    'tools_import_tree': 'this file; a developer tool, run by hand',
    'tools_huge_dv': 'diagnostic for the [HUGE-DV] warning, run by hand',
    'tools_gate_stability': 'confidence-gate margins and run-to-run '
                            'comparison, run by hand',
}


def _modules(root='.'):
    out = {}
    for dp, dirs, fs in os.walk(root):
        dirs[:] = [d for d in dirs if d not in ('__pycache__', '.git')]
        for f in fs:
            if not f.endswith('.py'):
                continue
            rel = os.path.relpath(os.path.join(dp, f), root)[:-3]
            name = rel.replace(os.sep, '.')
            if name.endswith('.__init__'):
                name = name[:-len('.__init__')]
            out[name] = os.path.join(dp, f)
    return out


def _deps(path, mods, package):
    found = set()
    for n in ast.walk(ast.parse(open(path, encoding='utf-8').read())):
        cands = []
        if isinstance(n, ast.Import):
            cands = [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            # `from . import x` -> base is the package; `from ._common import
            # y` -> base is package._common. Getting this wrong makes a module
            # that IS imported look dead, which is the one mistake this tool
            # must not make.
            base = (f'{package}.{n.module}' if n.level and n.module
                    else package if n.level else (n.module or ''))
            cands = [base] + [f'{base}.{a.name}' if base else a.name
                              for a in n.names]
        for c in cands:
            if c in mods:
                found.add(c)
            elif c.rsplit('.', 1)[0] in mods:
                found.add(c.rsplit('.', 1)[0])
    return found


def main():
    mods = _modules()
    seen, queue = set(), ['main']
    while queue:
        m = queue.pop()
        if m in seen or m not in mods:
            continue
        seen.add(m)
        package = m.rsplit('.', 1)[0] if '.' in m else ''
        queue += list(_deps(mods[m], mods, package))

    rest = sorted(set(mods) - seen)
    print(f'reachable from main.py ({len(seen)}):')
    for m in sorted(seen):
        print(f'    {m}')
    print(f'\nnot imported ({len(rest)}):')
    unexplained = []
    for m in rest:
        why = next((v for k, v in RUN_NOT_IMPORTED.items()
                    if m == k or m.startswith(k + '.')), None)
        print(f'    {m:<44} {why or "*** UNEXPLAINED ***"}')
        if why is None:
            unexplained.append(m)
    if unexplained:
        print(f'\n{len(unexplained)} module(s) neither imported nor explained. '
              f'Delete them or say why they are here.')
        return 1
    print('\nEverything present is either reachable or accounted for.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
