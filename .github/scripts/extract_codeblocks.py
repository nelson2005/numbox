#!/usr/bin/env python3
"""Lint Python code blocks inside .rst / .md docs with flake8.

Extracts every ``.. code-block:: python`` (or ``.. code:: python``) from .rst
files and every ```` ```python ```` fenced block from .md files, then passes
each block to ``flake8`` via stdin with ``--stdin-display-name=<path>:<line>``
so any reported violation maps back to the original source location.

Usage:
    extract_codeblocks.py PATH [PATH ...] [--max-line-length N] [--flake8 EXE]

Each PATH may be a file or a directory; directories are walked recursively.
Exits with the worst flake8 return code seen (0 if all clean, non-zero otherwise).
"""
import argparse
import re
import subprocess
import sys
import textwrap
from pathlib import Path


RST_DIRECTIVE = re.compile(r'^([ \t]*)\.\.\s+(?:code-block|code)::\s*python\b')
MD_FENCE = re.compile(r'^([ \t]*)```python\s*$')


def extract_rst_blocks(text):
    lines = text.splitlines(keepends=True)
    n = len(lines)
    i = 0
    while i < n:
        m = RST_DIRECTIVE.match(lines[i].rstrip('\n'))
        if not m:
            i += 1
            continue
        directive_indent = len(m.group(1))
        j = i + 1
        while j < n:
            ln = lines[j].rstrip('\n')
            if not ln.strip():
                j += 1
                continue
            this_indent = len(ln) - len(ln.lstrip())
            if this_indent > directive_indent and ln.lstrip().startswith(':'):
                j += 1
                continue
            break
        body_start = None
        body_indent = None
        body = []
        while j < n:
            ln = lines[j]
            stripped = ln.rstrip('\n')
            if not stripped.strip():
                if body_start is not None:
                    body.append(ln)
                j += 1
                continue
            this_indent = len(stripped) - len(stripped.lstrip())
            if body_start is None:
                if this_indent > directive_indent:
                    body_start = j + 1
                    body_indent = this_indent
                    body.append(ln)
                    j += 1
                else:
                    break
            else:
                if this_indent < body_indent:
                    break
                body.append(ln)
                j += 1
        while body and not body[-1].strip():
            body.pop()
        if body:
            yield body_start, textwrap.dedent(''.join(body))
        i = max(j, i + 1)


def extract_md_blocks(text):
    lines = text.splitlines(keepends=True)
    n = len(lines)
    i = 0
    while i < n:
        m = MD_FENCE.match(lines[i].rstrip('\n'))
        if not m:
            i += 1
            continue
        body_start = i + 2
        body = []
        i += 1
        while i < n:
            if lines[i].rstrip('\n').strip() == '```':
                break
            body.append(lines[i])
            i += 1
        i += 1
        if body:
            yield body_start, textwrap.dedent(''.join(body))


SKIP_DIRS = {'venv', '.venv', 'venv313', '_build', '.doctrees', 'node_modules', '.git', '__pycache__', 'plans'}


def _walk(root, suffix):
    for sub in sorted(root.rglob(f'*{suffix}')):
        if any(part in SKIP_DIRS for part in sub.parts):
            continue
        yield sub


def iter_files(paths):
    for raw in paths:
        p = Path(raw)
        if p.is_file():
            kind = {'.rst': 'rst', '.md': 'md'}.get(p.suffix)
            if kind:
                yield p, kind
        elif p.is_dir():
            for sub in _walk(p, '.rst'):
                yield sub, 'rst'
            for sub in _walk(p, '.md'):
                yield sub, 'md'


DOC_SNIPPET_IGNORE = 'E302,E303,E305,W292,W391'


def lint_block(code, label, max_line_length, extend_ignore, flake8_exe):
    proc = subprocess.run(
        [flake8_exe, f'--max-line-length={max_line_length}',
         f'--extend-ignore={extend_ignore}',
         f'--stdin-display-name={label}', '-'],
        input=code, text=True, capture_output=True,
    )
    if proc.stdout:
        sys.stdout.write(proc.stdout)
    if proc.stderr:
        sys.stderr.write(proc.stderr)
    return proc.returncode


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('paths', nargs='+')
    ap.add_argument('--max-line-length', type=int, default=120)
    ap.add_argument('--extend-ignore', default=DOC_SNIPPET_IGNORE,
                    help=f'flake8 codes to ignore beyond defaults (default: {DOC_SNIPPET_IGNORE} — '
                         'blank-line rules that do not apply to short snippets). Pass empty string to disable.')
    ap.add_argument('--flake8', default='flake8')
    args = ap.parse_args(argv)

    worst = 0
    any_blocks = False
    for path, kind in iter_files(args.paths):
        text = path.read_text(encoding='utf-8')
        blocks = extract_rst_blocks(text) if kind == 'rst' else extract_md_blocks(text)
        for start_line, code in blocks:
            any_blocks = True
            label = f'{path}:{start_line}'
            rc = lint_block(code, label, args.max_line_length, args.extend_ignore, args.flake8)
            if rc > worst:
                worst = rc
    if not any_blocks:
        print('extract_codeblocks: no Python code blocks found in given paths', file=sys.stderr)
    return worst


if __name__ == '__main__':
    sys.exit(main())
