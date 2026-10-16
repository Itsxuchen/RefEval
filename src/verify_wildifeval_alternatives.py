"""Independent masked-source verification of all saved WildIFEval channels.

No production ingestion, survey or joining helper is imported. Numeric literals
are immediately replaced by validity sentinels; no label threshold is applied.
"""
import argparse,json,hashlib,unicodedata
from pathlib import Path
from decimal import Decimal
from collections import defaultdict
PRODUCERS=('deepseek-v3','llama3.1-405b','llama3.3-70b','mistral-large','qwen2.5-72b')
VALID=object();INVALID=object()
def num(s):
    x=Decimal(s)
    return VALID if x.is_finite() and 0<=x<=1 else INVALID
def unique(pairs):
    d={}
    for k,v in pairs:
        if k in d:raise ValueError('Duplicate key in source')
        d[k]=v
    return d
def load(p):
    with p.open() as f:return json.load(f,parse_float=num,parse_int=num,parse_constant=lambda _:INVALID,object_pairs_hook=unique)
def sha(t):return hashlib.sha256(t.encode()).hexdigest()
def fsha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
    return h.hexdigest()
def norm(s):return ' '.join(unicodedata.normalize('NFC',s).split())


def verify(sources, alternatives, report_path, output_path):
    SRC,ALT,RPATH=Path(sources),Path(alternatives),Path(report_path)
    report=json.loads(RPATH.read_text())
    if report['source_manifest_sha256']!=fsha(SRC/'manifest.json') or report['alternative_manifest_sha256']!=fsha(ALT/'manifest.json') or report['inventory_sha256']!=fsha(ALT/'hf_inventory.json'):
        raise ValueError('Survey report does not bind current pinned source manifests')
    main=json.loads((SRC/'manifest.json').read_text());extra=json.loads((ALT/'manifest.json').read_text())
    checked=0
    for folder,manifest in [(SRC,main),(ALT,extra)]:
        for entry in manifest['files']:
            assert fsha(folder/entry['path'])==entry['sha256'];checked+=1
    inv=json.loads((ALT/'hf_inventory.json').read_text())
    assert inv['sha']==main['revisions']['predictions']==extra['revision']
    channels={}
    for entry in inv['siblings']:
        path=entry['rfilename']
        for producer in PRODUCERS:
            suffix='.'+producer+'-0shot-wild-if-eval.json'
            if path.startswith('llm_aaj_scores/') and path.endswith(suffix):
                stem=path[:-len(suffix)];name=stem.rsplit('/',1)[1]
                assert name.startswith('llm-aaj-')
                judge=name[len('llm-aaj-'):]
                run='agreement' if '/other_judges/' in path else 'main'
                channel=run+'/'+judge
                channels.setdefault(channel,{})[producer]=path
    assert set(channels)==set(report['channel_definitions'])
    scores={};outputs={}
    for producer in PRODUCERS:
        scores[producer]={}
        for channel,files in channels.items():
            file=SRC/('predictions/'+files[producer])
            if not file.exists():file=ALT/files[producer]
            data=load(file);compact={}
            for prompt,obj in data.items():
                vals=obj.get('scores') if isinstance(obj,dict) else None
                compact[prompt]=(frozenset(vals) if isinstance(vals,dict) else frozenset(),isinstance(vals,dict) and all(v is VALID for v in vals.values()))
            scores[producer][channel]=compact
        raw=load(SRC/f'predictions/model_predictions/{producer}-0shot-wild-if-eval.json')
        if 'predictions_key' in raw:
            val=raw['predictions_key'];raw=raw[val] if isinstance(val,str) else val
        outputs[producer]={}
        for prompt,msgs in raw.items():
            if not isinstance(msgs,list):continue
            users=[m.get('content') for m in msgs if isinstance(m,dict) and m.get('role')=='user']
            replies=[m.get('content') for m in msgs if isinstance(m,dict) and m.get('role')=='assistant']
            valid=len(users)==1 and users[0]==prompt and len(replies)==1 and isinstance(replies[0],str) and bool(replies[0].strip())
            outputs[producer][prompt]=(valid,sha(replies[0]) if len(replies)==1 and isinstance(replies[0],str) else '',sha(norm(prompt)))
    import pyarrow.parquet as pq
    bench=pq.read_table(SRC/'benchmark/data/test-00000-of-00001.parquet',columns=['conversation_id','decomposition']).to_pylist()
    tasks={sha('wildifeval-task|'+r['conversation_id']):tuple(r['decomposition']) for r in bench}
    signatures=defaultdict(set)
    for tid,cs in tasks.items():signatures[frozenset(cs)].add(tid)
    possible=defaultdict(set)
    for producer in PRODUCERS:
        for channel in channels:
            for prompt,(keys,valid) in scores[producer][channel].items():possible[prompt].update(signatures.get(keys,set()))
    linked=defaultdict(set)
    for prompt,tids in possible.items():
        if len(tids)==1:linked[next(iter(tids))].add(prompt)
    eligible={c:{p:set() for p in PRODUCERS} for c in channels}
    parent={t:t for t in tasks}
    def find(t):
        while parent[t]!=t:
            parent[t]=parent[parent[t]];t=parent[t]
        return t
    def merge(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    tokens={}
    for tid,cs in tasks.items():
        prompt=next(iter(linked[tid])) if len(linked[tid])==1 else None
        proper=prompt is not None and len(signatures[frozenset(cs)])==1 and len(set(cs))==len(cs) and 2<=len(cs)<=8
        for producer in PRODUCERS:
            validout,oh,ph=outputs[producer].get(prompt,(False,'',sha(norm(prompt)) if prompt is not None else ''))
            for kind,h in [('output',oh),('prompt',ph)]:
                if h:
                    token=(kind,h)
                    if token in tokens:merge(tid,tokens[token])
                    else:tokens[token]=tid
            for channel in channels:
                keys,valid=scores[producer][channel].get(prompt,(frozenset(),False))
                if proper and validout and valid and keys==set(cs):eligible[channel][producer].add(tid)
    groups=defaultdict(set)
    for tid in tasks:groups[find(tid)].add(tid)
    cluster={t:sha('wildifeval-cluster|'+'|'.join(sorted(ts))) for ts in groups.values() for t in ts}
    def counts(common):
        cls={cluster[t] for t in common};n=len(cls);pilot=n//5
        return {'common_tasks':len(common),'clusters':n,'pilot_clusters':pilot,'confirmation_clusters':n-pilot,
                'criteria_per_producer':sum(len(tasks[t]) for t in common),
                'coverage_feasible':n>=500 and pilot>=100 and n-pilot>=300 and len(common)/len(tasks)>=.7}
    channel_rows={}
    for channel in channels:
        common=set.intersection(*eligible[channel].values());row=counts(common)
        for key,value in row.items():assert report['channel_coverage'][channel][key if key!='criteria_per_producer' else 'criteria_per_producer_in_common_frame']==value,(channel,key)
        channel_rows[channel]=row
    combo_rows=[]
    for reported in report['combinations']:
        combo=reported['channels'];common=set.intersection(*(eligible[c][p] for c in combo for p in PRODUCERS));row=counts(common)
        for key,value in row.items():assert reported[key]==value,(combo,key)
        distinct=len({c.split('/',1)[1] for c in combo})==len(combo)
        assert reported['usable_under_original_coverage_gates']==(distinct and row['coverage_feasible'])
        combo_rows.append({'channels':combo,**row,'distinct_judges':distinct})
    receipt={'status':'PASS','metadata_only':True,'numeric_scores_retained':False,'threshold_applied':False,
             'production_join_helpers_imported':False,'source_files_hash_checked':checked,
             'channels_independently_checked':len(channel_rows),'combinations_independently_checked':len(combo_rows),
             'channel_counts':channel_rows,'top_distinct_judge_combinations':sorted([x for x in combo_rows if x['distinct_judges']],key=lambda r:r['common_tasks'],reverse=True)[:5],
             'survey_report_sha256':fsha(RPATH),'verifier_source_sha256':fsha(Path(__file__))}
    output=Path(output_path)
    if output.exists():raise ValueError("Preserve existing independent receipt")
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(receipt,indent=2)+"\n")
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources",type=Path,required=True)
    parser.add_argument("--alternatives",type=Path,required=True)
    parser.add_argument("--report",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result=verify(args.sources,args.alternatives,args.report,args.output)
    print(json.dumps({k:result[k] for k in ("status","metadata_only","source_files_hash_checked","channels_independently_checked","combinations_independently_checked")},indent=2))


if __name__=="__main__":
    main()
