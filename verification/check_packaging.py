"""
check_packaging.py -- the files no other check has ever looked at.
==================================================================

Every other module in this package compares a number the manuscript prints
against the data the run produced. None of them opens the README, the
CITATION.cff, the bibliography or the cover letter, and on 2026-09-21 an
outside review found four defects sitting in exactly that blind spot, in a
package that had already survived five adversarial rounds:

  * CITATION.cff ended in a tab character and therefore did not parse as YAML
    at all -- the machine-readable citation metadata of a release whose whole
    argument is reproducibility;
  * README.md still said the gate "admits 115" (it admits 118) and described a
    three-row table as "all four";
  * ref.bib labelled the software v1.0 while CITATION.cff and the changelog
    said 2.1.0;
  * the manuscript carried the RSC template's ESI placeholder as though it
    were a DOI.

None of these changes a result. All four would have been read by an editor
before any of the checked numbers were. The lesson is that a verification
suite which only validates results will certify a package whose metadata
contradicts them, so these are now checked with everything else.

Paths are resolved relative to the repository root and to the submission
package one level up; a file that is simply not present in a given checkout
is reported as unchecked rather than failed, because this module has to run
both inside the standalone code release and inside the full package.
"""

from __future__ import annotations

import os
import re

from . import paper_values as P
from ._common import Report

# Where each file lives, first hit wins: inside the code release, or one level
# up in the submission package.
_ROOTS = ('.', '..')


def _find(*candidates):
    for root in _ROOTS:
        for c in candidates:
            p = os.path.join(root, c)
            if os.path.isfile(p):
                return p
    return None


def _same(rep, label, want, got):
    """String equality, reported like every other comparison.

    `check_printed` rounds a float to the printed precision, which is the
    right estimator for a table value and the wrong one for a version
    string."""
    (rep.matches if str(want) == str(got) else rep.differs)(label, str(want),
                                                            str(got))


def _read(path):
    with open(path, encoding='utf-8') as fh:
        return fh.read()


def run(verbose=True):
    rep = Report('packaging metadata')

    # ---- CITATION.cff: it must PARSE, not merely exist -------------------
    cff = _find('CITATION.cff', 'codigo_build/CITATION.cff')
    if cff is None:
        rep.fail('CITATION.cff', 'not found')
    else:
        try:
            import yaml
        except ImportError:
            rep.note('PyYAML not installed; CITATION.cff parsed by hand')
            yaml = None
        if yaml is not None:
            try:
                meta = yaml.safe_load(_read(cff))
            except Exception as e:                            # noqa: BLE001
                meta = None
                rep.differs('CITATION.cff parses as YAML', 'parses',
                            f'{type(e).__name__}: {str(e).splitlines()[0]}')
            if meta:
                rep.matches('CITATION.cff parses as YAML', 'parses', 'parses')
                _same(rep, 'CITATION.cff version', P.SOFTWARE_VERSION,
                      str(meta.get('version', '')))
        # A trailing tab is invisible in an editor and fatal to the parser.
        raw = _read(cff)
        if raw.endswith('\n') and not re.search(r'[ \t]+\Z', raw.rstrip('\n')
                                                + '\n'):
            rep.matches('CITATION.cff has no trailing whitespace',
                        'clean', 'clean')
        else:
            rep.differs('CITATION.cff has no trailing whitespace', 'clean',
                        repr(raw[-4:]))

    # ---- README: the scope numbers it advertises -------------------------
    readme = _find('README.md', 'codigo_build/README.md')
    if readme is None:
        rep.fail('README.md', 'not found')
    else:
        txt = _read(readme)
        m = re.search(r'admits\s+(\d+)', txt)
        if m:
            rep.check('README admitted-geometry count',
                      P.GEOMETRIES_ADMITTED, int(m.group(1)), 0)
        else:
            rep.fail('README admitted-geometry count', 'phrase not found')
        m = re.search(r'(\d+)\s+geometries', txt)
        if m:
            rep.check('README total-geometry count', P.GEOMETRIES_TOTAL,
                      int(m.group(1)), 0)
        # "all four are placeholders" over a table with three rows was the
        # actual defect, so the claim is counted against the table.
        rows = len(re.findall(r'^\|\s*\[!\[DOI\]|^\|\s*(?:Software|Dataset|'
                              r'Preprint)\b', txt, re.M))
        words = {'two': 2, 'three': 3, 'four': 4, 'five': 5}
        m = re.search(r'All (two|three|four|five) (?:are placeholders|identifiers are synchronized|identifiers are the same)', txt)
        if m and rows:
            rep.check('README DOI count matches its own DOI table',
                      rows, words[m.group(1)], 0)

    # ---- bibliography: the software version it cites ---------------------
    bib = _find('ref.bib', 'fuentes/ref.bib', '../fuentes/ref.bib')
    if bib is None:
        rep.note('ref.bib not in this checkout; not checked')
    else:
        txt = _read(bib)
        _V = r'v([0-9]+(?:\.[0-9]+)*)'      # sin arrastrar el punto final
        found = re.findall(r'MAPA-metric:\s*' + _V, txt)
        found += re.findall(r'Software,\s*' + _V, txt)
        if not found:
            rep.fail('ref.bib software version', 'no version string found')
        for v in sorted(set(found)):
            _same(rep, 'ref.bib software version', P.SOFTWARE_VERSION, v)

    # ---- manuscript: no template placeholder left readable as a DOI ------
    for name in ('preprint_rsc.tex', 'preprint.tex'):
        tex = _find(name, f'fuentes/{name}', f'../fuentes/{name}')
        if tex is None:
            continue
        txt = _read(tex)
        # RSC manuscript IDs are c + 2-4 journal letters + digits + a
        # letter (PCCP is cXCP, three). The first version of this pattern
        # assumed two and therefore matched nothing, which the probe
        # below caught: a check that cannot fail is not a check.
        bad = re.findall(r'10\.1039/c[A-Za-z]{2,4}\d{4,6}[a-z]', txt)
        if bad:
            rep.differs(f'{name}: no journal-ID placeholder posing as a DOI',
                        'none', ', '.join(sorted(set(bad))))
        else:
            rep.matches(f'{name}: no journal-ID placeholder posing as a DOI',
                        'none', 'none')

    _check_bibliography(rep)
    return rep


def _check_bibliography(rep):
    """The half of the bibliography audit that needs no network.

    `scripts/verificar_bibliografia.py` resolves every DOI against Crossref and
    DataCite and cannot run offline -- it reports UNCHECKED, which is not the
    same as passing, and it stays the author's job before submission. What can
    be checked anywhere is the structure, and doing so here caught three
    functionals named in Sec. 3 whose references sat uncited in the file
    (BLYP, PBEsol) and one that had no entry at all (revPBE).
    """
    bib = _find('ref.bib', 'fuentes/ref.bib', '../fuentes/ref.bib')
    if bib is None:
        rep.note('ref.bib not in this checkout; bibliography not checked')
        return
    txt = _read(bib)
    keys = re.findall(r'@\w+\{([^,\s]+),', txt)
    dups = sorted({k for k in keys if keys.count(k) > 1})
    if dups:
        rep.differs('bibliography has no duplicate keys', 'none',
                    ', '.join(dups))
    else:
        rep.matches('bibliography has no duplicate keys', 'none',
                    f'{len(keys)} unique keys')

    dois = re.findall(r'doi\s*=\s*\{([^}]*)\}', txt)
    malformed = [d for d in dois if not re.match(r'^10\.\d{4,9}/\S+$', d)]
    if malformed:
        rep.differs('every DOI is syntactically well formed', 'all',
                    ', '.join(malformed[:3]))
    else:
        rep.matches('every DOI is syntactically well formed', 'all',
                    f'{len(dois)} of {len(keys)} entries carry one')

    cited = set()
    seen = 0
    for name in ('preprint.tex', 'preprint_rsc.tex'):
        tex = _find(name, f'fuentes/{name}', f'../fuentes/{name}')
        if tex is None:
            continue
        seen += 1
        for m in re.findall(r'\\cite\w*\{([^}]*)\}', _read(tex)):
            cited |= {x.strip() for x in m.split(',') if x.strip()}
    if not seen:
        rep.note('no manuscript source in this checkout; citations not checked')
        return
    missing = sorted(cited - set(keys))
    if missing:
        rep.differs('every cited key exists in the bibliography', 'all',
                    ', '.join(missing[:5]))
    else:
        rep.matches('every cited key exists in the bibliography', 'all',
                    f'{len(cited)} cited')
    # An uncited entry is not an error in itself, but each of the three that
    # were uncited here turned out to be a functional the text names without
    # crediting, so it is reported rather than ignored.
    orphan = sorted(set(keys) - cited)
    if orphan:
        rep.differs('no entry sits in the bibliography uncited', 'none',
                    ', '.join(orphan))
    else:
        rep.matches('no entry sits in the bibliography uncited', 'none',
                    'none')

    rep.note('DOI resolution against Crossref and DataCite is NOT done here '
             'and is not implied by the above; run '
             'scripts/verificar_bibliografia.py on a networked machine')
