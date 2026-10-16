"""Outcome-blind survey of every saved judge channel for five WildIFEval producers.

This extends the failed preflight; it never changes the original source cache,
protocol, model roles, or gates. A saved grading run is a channel, not necessarily
an independent judge identity. Numeric JSON literals become validity sentinels.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import itertools
import json
from pathlib import Path
import subprocess

from src.verify_wildifeval_preflight import (
    PINNED_REVISIONS, PRODUCERS, key_diagnostics, normalized,
    read_output_metadata, read_score_metadata, sha_file, sha_text, validate_inventory,
)

PREDICTION_REPO = 'gililior/wild-if-eval-predictions'


def discover_channels(inventory):
    if inventory.get('sha') != PINNED_REVISIONS['predictions']:
        raise ValueError('Inventory revision differs from pinned prediction revision')
    names = [x['rfilename'] for x in inventory['siblings']]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate inventory filename')
    channels = {}
    for name in sorted(names):
        if not name.startswith('llm_aaj_scores/') or not name.endswith('-0shot-wild-if-eval.json'):
            continue
        for producer in PRODUCERS:
            suffix = '.' + producer + '-0shot-wild-if-eval.json'
            if not name.endswith(suffix):
                continue
            stem = name[:-len(suffix)]
            folder, base = stem.rsplit('/', 1)
            if not base.startswith('llm-aaj-'):
                raise ValueError('Unexpected grader filename')
            judge = base[len('llm-aaj-'):]
            run = 'agreement' if folder == 'llm_aaj_scores/other_judges' else 'main'
            if folder not in ('llm_aaj_scores', 'llm_aaj_scores/other_judges'):
                raise ValueError('Unexpected grader directory')
            channel = run + '/' + judge
            entry = channels.setdefault(channel, {'judge': judge, 'run': run, 'files': {}})
            if producer in entry['files']:
                raise ValueError('Ambiguous producer file in a channel')
            entry['files'][producer] = name
            break
    return channels


def validate_original_sources(sources):
    manifest = json.loads((Path(sources) / 'manifest.json').read_text())
    validate_inventory(manifest)
    for entry in manifest['files']:
        if sha_file(Path(sources) / entry['path']) != entry['sha256']:
            raise ValueError('Original source hash differs')
    return manifest


def _validate_extra_manifest(directory, manifest):
    if manifest['revision'] != PINNED_REVISIONS['predictions']:
        raise ValueError('Alternative snapshot revision mismatch')
    paths = [f['path'] for f in manifest['files']]
    if len(paths) != len(set(paths)):
        raise ValueError('Duplicate alternative source manifest paths')
    for item in manifest['files']:
        path = Path(item['path'])
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Invalid alternative source path')
        actual = Path(directory) / path
        if actual.stat().st_size != item['bytes'] or sha_file(actual) != item['sha256']:
            raise ValueError('Alternative source hash differs: ' + item['path'])
    return manifest


def download_alternatives(sources: Path, directory: Path):
    """Download only missing fixed-revision channels into a separate cache."""
    sources, directory = Path(sources), Path(directory)
    original = validate_original_sources(sources)
    original_paths = {x['path'] for x in original['files']}
    directory.mkdir(parents=True, exist_ok=True)
    existing_manifest = directory / 'manifest.json'
    if existing_manifest.exists():
        _validate_extra_manifest(directory, json.loads(existing_manifest.read_text()))
    inventory_path = directory / 'hf_inventory.json'
    if not inventory_path.exists():
        url = f'https://huggingface.co/api/datasets/{PREDICTION_REPO}/revision/{PINNED_REVISIONS["predictions"]}'
        tmp = directory / 'hf_inventory.json.partial'
        subprocess.run(['curl', '-fsSL', '--retry', '2', '--connect-timeout', '20', '--max-time', '90', url, '-o', str(tmp)], check=True)
        discover_channels(json.loads(tmp.read_text()))
        tmp.rename(inventory_path)
    inventory = json.loads(inventory_path.read_text())
    channels = discover_channels(inventory)
    needed = sorted({name for channel in channels.values() for name in channel['files'].values()
                     if 'predictions/' + name not in original_paths})
    for name in needed:
        out = directory / name
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.exists():
            tmp = out.with_suffix(out.suffix + '.partial')
            url = f'https://huggingface.co/datasets/{PREDICTION_REPO}/resolve/{PINNED_REVISIONS["predictions"]}/{name}'
            subprocess.run(['curl', '-fsSL', '--retry', '2', '--connect-timeout', '20', '--max-time', '300', url, '-o', str(tmp)], check=True)
            tmp.rename(out)
    entries = [{'path': name, 'bytes': (directory / name).stat().st_size, 'sha256': sha_file(directory / name)}
               for name in ['hf_inventory.json'] + needed]
    result = {'schema_version': 1, 'repo_id': PREDICTION_REPO, 'revision': PINNED_REVISIONS['predictions'],
              'files': entries, 'new_score_files': len(needed), 'original_sources_manifest_sha256': sha_file(sources / 'manifest.json'),
              'scope': 'Pinned repository inventory and previously uncached grader files; original 28-file cache unchanged'}
    if existing_manifest.exists() and json.loads(existing_manifest.read_text()) != result:
        raise ValueError('Preserve existing alternatives manifest; snapshot changed')
    existing_manifest.write_text(json.dumps(result, indent=2) + '\n')
    return result


def coverage_gates(common, clusters, benchmark_count):
    ordered = sorted(set(clusters), key=lambda c: sha_text('refeval-external-v1|' + c))
    pilot = ordered[:len(ordered) // 5]
    retention = len(common) / benchmark_count if benchmark_count else 0.0
    gates = {'clusters_at_least_500': len(ordered) >= 500, 'pilot_clusters_at_least_100': len(pilot) >= 100,
             'confirmation_clusters_at_least_300': len(ordered) - len(pilot) >= 300,
             'retention_each_at_least_70pct': retention >= .7}
    return {'common_tasks': len(common), 'clusters': len(ordered), 'pilot_clusters': len(pilot),
            'confirmation_clusters': len(ordered) - len(pilot), 'retention_each_producer': retention,
            'gates': gates, 'coverage_feasible': all(gates.values()),
            'common_task_ids_sha256': sha_text('|'.join(sorted(common))),
            'common_cluster_ids_sha256': sha_text('|'.join(ordered))}


def survey_metadata(benchmark, outputs, scores, channels, producers=PRODUCERS):
    """Join only complete literal criterion sets; never score, align or drop bits."""
    tasks, signatures = {}, defaultdict(set)
    for row in benchmark:
        conv, criteria = row['conversation_id'], tuple(row['decomposition'])
        if not isinstance(conv, str) or not all(isinstance(c, str) for c in criteria):
            raise ValueError('Invalid benchmark schema')
        task_id = sha_text('wildifeval-task|' + conv)
        if task_id in tasks:
            raise ValueError('Duplicate benchmark conversation ID')
        tasks[task_id] = criteria
        signatures[tuple(sorted(criteria))].add(task_id)
    candidates = defaultdict(set)
    for producer in producers:
        for channel in channels:
            for prompt, metadata in scores[producer][channel].items():
                candidates[prompt].update(signatures.get(tuple(sorted(metadata['keys'])), ()))
    prompts_by_task = defaultdict(set)
    for prompt, task_ids in candidates.items():
        if len(task_ids) == 1:
            prompts_by_task[next(iter(task_ids))].add(prompt)
    eligible = {c: {p: set() for p in producers} for c in channels}
    token_tasks = defaultdict(set)
    for task_id, criteria in tasks.items():
        prompts = prompts_by_task[task_id]
        prompt = next(iter(prompts)) if len(prompts) == 1 else None
        proper = (len(signatures[tuple(sorted(criteria))]) == 1 and len(criteria) == len(set(criteria))
                  and 2 <= len(criteria) <= 8 and prompt is not None)
        for producer in producers:
            output = outputs[producer].get(prompt, {'valid': False, 'output_hash': '', 'normalized_prompt_hash': ''})
            prompt_hash = output['normalized_prompt_hash'] or (sha_text(normalized(prompt)) if prompt is not None else '')
            for kind, value in [('prompt', prompt_hash), ('output', output['output_hash'])]:
                if value:
                    token_tasks[(kind, value)].add(task_id)
            for channel in channels:
                metadata = scores[producer][channel].get(prompt)
                if (proper and output['valid'] and metadata is not None and metadata['valid_object']
                        and metadata['keys'] == set(criteria) and metadata['valid_keys'] == set(criteria)):
                    eligible[channel][producer].add(task_id)
    neighbors = {t: set() for t in tasks}
    for members in token_tasks.values():
        if len(members) > 1:
            first = min(members)
            neighbors[first].update(members - {first})
            for task_id in members - {first}:
                neighbors[task_id].add(first)
    unseen, components, clusters = set(tasks), [], {}
    while unseen:
        pending = [unseen.pop()]; members = set(pending)
        while pending:
            current = pending.pop(); new = neighbors[current] & unseen
            unseen.difference_update(new);members.update(new);pending.extend(new)
        components.append(members)
        cluster = sha_text('wildifeval-cluster|' + '|'.join(sorted(members)))
        clusters.update({task_id: cluster for task_id in members})
    channel_coverage = {}
    for channel in channels:
        common = set.intersection(*(eligible[channel][p] for p in producers))
        channel_coverage[channel] = {'judge': channels[channel]['judge'], 'run': channels[channel]['run'],
                                    'source_task_entries_by_producer': {p: len(scores[p][channel]) for p in producers},
                                    'complete_current_benchmark_tasks_by_producer': {p: len(eligible[channel][p]) for p in producers},
                                    'criteria_per_producer_in_common_frame': sum(len(tasks[t]) for t in common),
                                    'criteria_all_producers_in_common_frame': len(producers) * sum(len(tasks[t]) for t in common),
                                    **coverage_gates(common, {clusters[t] for t in common}, len(tasks))}
    combinations = []
    for size in (2, 3, 4):
        for combo in itertools.combinations(sorted(channels), size):
            common = set.intersection(*(eligible[c][p] for c in combo for p in producers))
            distinct = len({channels[c]['judge'] for c in combo}) == size
            summary = coverage_gates(common, {clusters[t] for t in common}, len(tasks))
            combinations.append({'channels': list(combo), 'distinct_judge_identities': distinct,
                                 'units': len(common) * len(producers), 'criteria_per_producer': sum(len(tasks[t]) for t in common), **summary,
                                 'usable_under_original_coverage_gates': distinct and summary['coverage_feasible']})
    pairs = []
    for a, b in itertools.combinations(sorted(channels), 2):
        producer_diagnostics = {}
        same_valid_prompts = []
        for producer in producers:
            left, right = scores[producer][a], scores[producer][b]
            diagnostic = key_diagnostics(left, right)
            shared = set(left) & set(right)
            compatible = {t for t in shared if (left[t]['valid_object'] and right[t]['valid_object']
                          and 2 <= len(left[t]['keys']) <= 8 and left[t]['keys'] == right[t]['keys']
                          and left[t]['valid_keys'] == left[t]['keys'] and right[t]['valid_keys'] == right[t]['keys'])}
            same_valid_prompts.append(compatible)
            diagnostic.update({'left_task_entries': len(left), 'right_task_entries': len(right),
                               'only_left_task_keys': len(set(left) - set(right)), 'only_right_task_keys': len(set(right) - set(left)),
                               'same_complete_valid_vector_k2to8': len(compatible)})
            producer_diagnostics[producer] = diagnostic
        pairs.append({'channels': [a, b], 'distinct_judge_identities': channels[a]['judge'] != channels[b]['judge'],
                      'by_producer': producer_diagnostics,
                      'common_same_complete_valid_raw_prompt_vectors_across_producers': len(set.intersection(*same_valid_prompts)),
                      'raw_shared_scope': 'Task-text keys across entire saved files; not necessarily released benchmark tasks'})
    return {'status': 'masked_alternative_coverage', 'schema_version': 1, 'numeric_scores_retained': False,
            'pass_fail_threshold_applied': False, 'outcome_analysis_performed': False,
            'benchmark_rows': len(tasks), 'producers': list(producers), 'saved_channels': len(channels),
            'distinct_judges': len({v['judge'] for v in channels.values()}),
            'channel_definitions': channels, 'channel_coverage': channel_coverage, 'combinations': combinations,
            'raw_pair_diagnostics': pairs, 'all_combinations_fail_original_coverage_gates': not any(c['coverage_feasible'] for c in combinations),
            'any_usable_combination': any(c['usable_under_original_coverage_gates'] for c in combinations),
            'total_clusters_before_complete_case': len(components),
            'largest_cluster_before_complete_case': max(map(len, components), default=0),
            'join_rule': 'Unique exact full current-benchmark criterion-set linkage, conservative union over all surveyed channels; no semantic matching',
            'cluster_rule': 'Connected conversation, normalized-prompt and output-hash duplicates using all linked benchmark metadata',
            'split_rule': 'Hypothetical original fixed 20/80 hash split applied per metadata frame; no pilot labels revealed',
            'output_link_limit': 'Judge files name producer and task text but do not embed the evaluated output hash; file association cannot independently prove output version',
            'scope_limit': 'All grader channels in the pinned inventory for the fixed five producers; not historical branches, unlisted assets or other producer models'}


def survey(sources, alternatives, output):
    sources, alternatives, output = Path(sources), Path(alternatives), Path(output)
    original = validate_original_sources(sources)
    extra = _validate_extra_manifest(alternatives, json.loads((alternatives / 'manifest.json').read_text()))
    if extra['original_sources_manifest_sha256'] != sha_file(sources / 'manifest.json'):
        raise ValueError('Alternative manifest does not bind current original cache')
    inventory = json.loads((alternatives / 'hf_inventory.json').read_text())
    channels = discover_channels(inventory)
    base_files = {x['path'] for x in original['files']}
    expected_extra = {'hf_inventory.json'} | {f for c in channels.values() for f in c['files'].values() if 'predictions/' + f not in base_files}
    if {x['path'] for x in extra['files']} != expected_extra:
        raise ValueError('Alternative manifest does not cover all missing saved channels')
    outputs, scores = {}, {}
    for producer in PRODUCERS:
        outputs[producer] = read_output_metadata(sources / f'predictions/model_predictions/{producer}-0shot-wild-if-eval.json')
        scores[producer] = {}
        for channel, metadata in channels.items():
            name = metadata['files'].get(producer)
            if name is None:
                scores[producer][channel] = {}
            else:
                path = sources / ('predictions/' + name) if 'predictions/' + name in base_files else alternatives / name
                scores[producer][channel] = read_score_metadata(path)
    import pyarrow.parquet as pq
    benchmark = pq.read_table(sources / 'benchmark/data/test-00000-of-00001.parquet', columns=['conversation_id', 'decomposition']).to_pylist()
    report = survey_metadata(benchmark, outputs, scores, channels)
    report.update({'source_revisions': PINNED_REVISIONS, 'source_manifest_sha256': sha_file(sources / 'manifest.json'),
                   'alternative_manifest_sha256': sha_file(alternatives / 'manifest.json'),
                   'inventory_sha256': sha_file(alternatives / 'hf_inventory.json'),
                   'survey_source_sha256': sha_file(Path(__file__)),
                   'masked_parser_source_sha256': sha_file(Path(__file__).with_name('verify_wildifeval_preflight.py')),
                   'source_files_checked': len(original['files']) + len(extra['files']),
                   'pinned_inventory_files': len(inventory['siblings'])})
    if output.exists():
        raise ValueError('Preserve previous survey report; choose a new output path')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('download', 'survey'))
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--alternatives', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.command == 'download':
        result = download_alternatives(args.sources, args.alternatives)
        print(json.dumps({'new_score_files': result['new_score_files'], 'revision': result['revision']}))
    else:
        if args.output is None:
            parser.error('--output required for survey')
        result = survey(args.sources, args.alternatives, args.output)
        print(json.dumps({k: result[k] for k in ('status', 'saved_channels', 'distinct_judges', 'any_usable_combination', 'all_combinations_fail_original_coverage_gates')}, indent=2))


if __name__ == '__main__':
    main()
