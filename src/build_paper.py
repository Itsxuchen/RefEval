"""Compile the complete manuscript with Tectonic; leave published artifacts intact."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import re
import zipfile
from pathlib import Path


def source_archive(root: Path, destination: Path) -> Path:
    """Package only the manuscript's required inputs, compilable from ZIP root."""
    root, destination = root.resolve(), destination.resolve()
    if destination.suffix != '.zip':
        raise ValueError('Source archive must have a .zip suffix')
    protected = ('data', 'src', 'tests', 'context', 'docs', 'artifacts/expected',
                 'artifacts/figures', 'artifacts/validation')
    if any(destination.is_relative_to(root / p) for p in protected):
        raise ValueError('Source archive cannot overwrite a protected tree')
    manuscript = root / 'artifacts/manuscript'
    paper = (manuscript / 'paper.tex').read_text()
    appendix = (manuscript / 'methods_appendix.tex').read_text()
    old_path = r'\graphicspath{{../figures/}}'
    if paper.count(old_path) != 1:
        raise ValueError('Expected exactly one manuscript figure-path declaration')
    entries = {'paper.tex': paper.replace(old_path, r'\graphicspath{{figures/}}').encode(),
               'methods_appendix.tex': appendix.encode(),
               'paper_references.bib': (manuscript / 'paper_references.bib').read_bytes()}
    figures = set(re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', paper + appendix))
    for figure in sorted(figures):
        if Path(figure).name != figure or not figure.endswith('.pdf'):
            raise ValueError(f'Expected a plain PDF figure name: {figure}')
        entries[f'figures/{figure}'] = (root / 'artifacts/figures' / figure).read_bytes()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(entries.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, content)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--tectonic', default='tectonic', help='Tectonic executable (validated with 0.17.0)')
    parser.add_argument('--output', type=Path, help='Build directory; defaults to artifacts/paper-build')
    parser.add_argument('--source-archive', type=Path, help='Optional portable source ZIP to create after compilation')
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
    archive = source_archive(root, args.source_archive) if args.source_archive else None
    print(json.dumps({'compiled': True, 'pdf': str(pdf),
                      'source_archive': str(archive) if archive else None,
                      'scope': 'LaTeX compilation; scientific claim validation is a separate command.'}, indent=2))


if __name__ == '__main__':
    main()
