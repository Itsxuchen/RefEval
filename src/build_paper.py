"""Compile the complete manuscript with Tectonic; leave published artifacts intact."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--tectonic', default='tectonic', help='Tectonic executable (validated with 0.17.0)')
    parser.add_argument('--output', type=Path, help='Build directory; defaults to artifacts/paper-build')
    args = parser.parse_args()
    root = args.package_root.resolve()
    source = root / 'artifacts/manuscript/paper.tex'
    out = (args.output or root / 'artifacts/paper-build').resolve()
    if not source.is_file():
        parser.error(f'Manuscript source missing: {source}')
    # A rebuild must not overwrite versioned sources, references or published PDFs.
    protected = ['data', 'src', 'tests', 'context', 'docs', 'artifacts/manuscript',
                 'artifacts/expected', 'artifacts/figures', 'artifacts/validation']
    if out == root or any(out.is_relative_to(root / p) or (root / p).is_relative_to(out)
                          for p in protected):
        parser.error('Choose a build directory outside protected input and release trees.')
    exe = shutil.which(args.tectonic)
    if exe is None:
        parser.error('Tectonic was not found. Install Tectonic 0.17.0 or pass --tectonic /path/to/tectonic.')
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run([exe, '-X', 'compile', source.name, '--outdir', str(out),
                    '--keep-logs', '--keep-intermediates'], cwd=source.parent, check=True)
    pdf = out / 'paper.pdf'
    if not pdf.is_file() or pdf.stat().st_size == 0:
        raise RuntimeError('Compiler completed without a nonempty paper.pdf')
    print(json.dumps({'compiled': True, 'pdf': str(pdf),
                      'scope': 'LaTeX compilation; scientific claim validation is a separate command.'}, indent=2))


if __name__ == '__main__':
    main()
