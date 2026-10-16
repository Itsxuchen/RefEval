"""Masked public ProfBench metadata audit; no task scores or model errors.

Human fulfilment fields are replaced during object construction by validity and
type metadata. Label values, task pass rates and agreement outcomes are never
returned. Source text remains only in the local download cache.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

MODELS = ('o3', 'grok4', 'r1-0528')
CODE_SHA = 'b06a29cda4d1433e9a9aad8171e4086299083e94'
DATA_SHA = 'f5c471cf2f277c264621e4b5c0cad5bb466f9260'
WEIGHTS = {'Critical': 4, 'Major': 3, 'Minor': 2, 'Additional': 1}
EXPECTED_INPUT_SHA256 = {
    'test.jsonl': 'c6b01278e67cb94acde66864ae2f4fb2af4b2d3c637491e9abf2190e00c56721',
    'utils.py': '6aa796c531e1601369cd7249e1f33b0a941e993ef90a7e9004170ee01665d3fe',
}


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def masked_object(pairs):
    row = {}
    for key,value in pairs:
        if key in row:
            raise ValueError('Duplicate JSON key in ProfBench source')
        if key.endswith('_fulfilment'):
            value = {'valid_binary': isinstance(value,(bool,int)) and value in (0,1),
                     'source_type':type(value).__name__}
        row[key]=value
    return row


def profile(rows, judge_template):
    task_ids = [r['task_id'] for r in rows]
    if len(task_ids)!=len(set(task_ids)):
        raise ValueError('Duplicate base-task identifier')
    domains=Counter(r['domain'] for r in rows)
    top_fields=sorted({k for r in rows for k in r})
    rubric_fields=sorted({k for r in rows for c in r['rubrics'] for k in c})
    sizes=[len(r['rubrics']) for r in rows]
    weights=Counter(c['criterion_weight'] for r in rows for c in r['rubrics'])
    types=Counter(str(t) for r in rows for c in r['rubrics'] for t in (
        c['criterion_type'] if isinstance(c['criterion_type'],list) else [c['criterion_type']]))
    invalid_weights=sum(n for w,n in weights.items() if w not in WEIGHTS)
    missing_labels=Counter()
    label_types=Counter()
    complete_units=0
    duplicate_descriptions=0
    units=[]
    input_chars=[]
    input_bytes=[]
    prompt_lengths=[]
    output_lengths=[]
    per_model=Counter()
    for task in rows:
        criteria=task['rubrics']
        descriptions=[c['criterion_description'] for c in criteria]
        duplicate_descriptions+=len(descriptions)-len(set(descriptions))
        prompt_lengths.append(len(task['prompt']))
        for model in MODELS:
            response=task[model+'_response']
            valid_response=isinstance(response,str) and bool(response.strip())
            flags=[]
            for c in criteria:
                label=c.get(model+'_fulfilment')
                valid=isinstance(label,dict) and label.get('valid_binary') is True
                flags.append(valid)
                missing_labels[model]+=int(not valid)
                label_types[label.get('source_type','missing') if isinstance(label,dict) else 'missing']+=1
                if valid_response:
                    prompt=judge_template.format(response=response,criterion_description=c['criterion_description'])
                    input_chars.append(len(prompt));input_bytes.append(len(prompt.encode('utf-8')))
            complete=valid_response and all(flags)
            complete_units+=complete
            per_model[model]+=len(criteria)
            output_lengths.append(len(response) if isinstance(response,str) else 0)
            units.append({'task_id':task['task_id'],'domain':task['domain'],'producer':model,'k':len(criteria),
                          'prompt_sha256':digest(task['prompt']),
                          'response_sha256':digest(response) if isinstance(response,str) else None,
                          'criterion_description_hashes':[digest(d) for d in descriptions],
                          'valid_response':valid_response,'complete_reference_mask':complete,
                          'response_characters':len(response) if isinstance(response,str) else 0})
    def stats(values):
        return {'count':len(values),'sum':sum(values),'min':min(values),'median':statistics.median(values),
                'mean':statistics.mean(values),'max':max(values)}
    return {
      'base_tasks':len(rows),'domains':dict(domains),'output_units':len(units),
      'criterion_definitions':sum(sizes),'response_criterion_pairs':sum(per_model.values()),
      'response_criterion_pairs_by_producer':dict(per_model),'rubric_size':stats(sizes),
      'rubric_size_histogram':dict(sorted(Counter(sizes).items())),
      'criterion_weights':dict(weights),'unknown_weight_entries':invalid_weights,
      'criterion_types':dict(types),'duplicate_descriptions_within_task':duplicate_descriptions,
      'top_level_fields':top_fields,'rubric_fields':rubric_fields,
      'released_annotator_id_fields':[x for x in top_fields+rubric_fields if any(s in x.lower() for s in ('annotator','rater','reviewer','adjudicat'))],
      'released_judge_prediction_fields':[x for x in top_fields+rubric_fields if any(s in x.lower() for s in ('judge','prediction','rating'))],
      'invalid_reference_fields_by_producer':dict(missing_labels),'reference_field_types':dict(label_types),
      'complete_reference_output_units':complete_units,
      'unique_prompt_hashes':len({u['prompt_sha256'] for u in units}),
      'unique_response_hashes':len({u['response_sha256'] for u in units}),
      'prompt_characters':stats(prompt_lengths),'response_characters':stats(output_lengths),
      'criterion_call_input_characters':stats(input_chars),'criterion_call_input_utf8_bytes':stats(input_bytes),
      'potential_calls_per_judge':len(input_chars),'unit_metadata':units,
    }


def analyze(source_dir,output):
    source_dir,output=Path(source_dir),Path(output)
    for name, expected in EXPECTED_INPUT_SHA256.items():
        if hashlib.sha256((source_dir/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Pinned ProfBench input hash mismatch: ' + name)
    metadata=json.loads((source_dir/'hf_metadata.json').read_text())
    commit=json.loads((source_dir/'github_commit.json').read_text())
    if metadata['sha']!=DATA_SHA or commit['sha']!=CODE_SHA:
        raise ValueError('Unexpected snapshot revision')
    rows=[]
    with (source_dir/'test.jsonl').open() as handle:
        for line in handle:
            rows.append(json.loads(line,object_pairs_hook=masked_object))
    # Read a constant from syntax; never import or execute downloaded code.
    tree=ast.parse((source_dir/'utils.py').read_text())
    templates=[ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
               and any(isinstance(t,ast.Name) and t.id=='QUESTION_PROMPT_YES_NO' for t in n.targets)]
    if len(templates)!=1 or not isinstance(templates[0],str):raise ValueError('Judge template missing/ambiguous')
    report={'status':'metadata_feasibility_only','reference_values_exposed':False,
            'mechanism_outcomes_computed':False,'model_calls':0,
            'code_revision':CODE_SHA,'dataset_revision':DATA_SHA,
            'metadata':profile(rows,templates[0])}
    code_tree=json.loads((source_dir/'github_tree.json').read_text())
    report['source_inventory']={'dataset_files':[s['rfilename'] for s in metadata['siblings']],
                                'github_files':[s['path'] for s in code_tree['tree'] if s['type']=='blob']}
    report['sources']={p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
                       for p in sorted(source_dir.iterdir()) if p.is_file()}
    report['primary_source_urls']=[
       f'https://huggingface.co/datasets/nvidia/ProfBench/tree/{DATA_SHA}',
       f'https://github.com/NVlabs/ProfBench/tree/{CODE_SHA}',
       'https://arxiv.org/html/2510.18941v1']
    report['interpretation']={
      'reference_provenance':'Paper describes domain professionals creating tasks/rubrics and scoring three outputs; reviewers approve stages. Public release has one binary fulfilment field per output-criterion, without rater IDs or repeat vectors.',
      'repeat_validation':'Paper reports 1,127 response-criterion pairs reannotated by two additional experts and Fleiss kappa .912; those individual repeat ratings are not among released fields. This does not establish unanimous/adjudicated truth for every public label.',
      'native_target':'Positive importance-weighted criterion fulfilment; weights Additional=1, Minor=2, Major=3, Critical=4. All-pass would be a constructed strict full-fulfilment target, not the native headline score.',
      'judge_availability':'No saved judge predictions in the pinned four-file HF release or nine-file GitHub tree. Existing human labels are not existing independent judge predictions.',
      'input_scope':'Official judge template receives the saved response and one criterion, not original task prompt, external documents, PDFs or images. Report-generation code supports documents but disables upload for the public release.',
      'sampling_scope':'Forty base-task clusters across four domains; 120 model-output units are dependent within tasks. A 20/80 split means eight pilot and 32 confirmation clusters, not 24/96 independent tasks.',
      'design_recommendation':'Useful small human-reference external replication if a new frozen judge run is authorized. Prefer a pre-specified aggregate replication with domain-stratified whole-task grouping; do not transplant the WildIFEval 500-cluster gate or claim a high-powered pilot forecast test.',
      'unknowns':'No task-pass prevalence or judge error composition checked; long rubrics may make strict passes rare. This is a prospective feasibility concern, not an observed outcome.',
      'license':'NVIDIA Evaluation Dataset License governs data; repository code MIT. Preserve local sources and publish download/hash instructions before considering redistribution.'}
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise ValueError('Preserve existing report')
    output.write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=analyze(args.sources,args.output)
    print(json.dumps({k:result['metadata'][k] for k in ('base_tasks','output_units','criterion_definitions','response_criterion_pairs','rubric_size','criterion_weights','complete_reference_output_units','potential_calls_per_judge')},indent=2))
