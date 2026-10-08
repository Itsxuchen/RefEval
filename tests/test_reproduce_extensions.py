"""Small real-table checks of extension verification, without scientific replay."""
import csv
import gzip
import io
import json
from pathlib import Path

import pytest

from src import reproduce_extensions as extension
from src.reproduce import validate_output


def write_table(path, rows, *, compressed_mtime=1):
    content=io.StringIO(newline="")
    writer=csv.DictWriter(content,fieldnames=["dataset","seed","estimate"])
    writer.writeheader();writer.writerows(rows)
    data=content.getvalue().encode()
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(gzip.compress(data,mtime=compressed_mtime) if path.name.endswith('.gz') else data)


@pytest.fixture
def replay_fixture(tmp_path,monkeypatch):
    root=tmp_path/'package'
    expected=root/'artifacts/expected'
    output=root/'artifacts/replayed-extensions'
    monkeypatch.setattr(extension,'ROOT',root)
    monkeypatch.setattr(extension,'EXPECTED',expected)
    monkeypatch.setattr(extension,'validate_output',lambda path:validate_output(path,root))
    plain=[dict(dataset='cell-A',seed=i,estimate=.1*i) for i in (1,2,3)]
    compressed=[dict(dataset='cell-B',seed=i,estimate=.2*i) for i in (1,2)]
    for folder in extension.DIRECTORIES:
        for target,mtime in [(expected,1),(output,99)]:
            write_table(target/folder/'counts.csv',plain)
            write_table(target/folder/'curves.csv.gz',compressed,compressed_mtime=mtime)
            # Run metadata is deliberately not a scientific comparison target.
            (target/folder/'manifest.json').write_text(json.dumps({'seconds':mtime,'source_paths':[str(target)],'source_sha256':str(mtime)}))
    for relative in extension.JSON_TARGETS:
        scientific={'dataset':'fixed-frame','rows':5,'rate':.25,'counts':[2,3],
                    'nested':{'seed':31,'units':206}}
        (expected/relative).write_text(json.dumps({**scientific,'created_utc':'before','source_hashes':{'old':'digest'},'seconds':3}))
        (output/relative).write_text(json.dumps({**scientific,'created_utc':'after','source_hashes':{'new':'digest'},'seconds':7}))
    return root,expected,output,plain,compressed


def test_frozen_configuration_and_scientific_target_scope():
    assert extension.CONFIG==dict(mechanism_permutations=64,order_seeds=31,
                                  cluster_replicates=399,cluster_seed=20261008,
                                  reference_cells=4,stratified_cells=9)
    assert set(extension.DIRECTORIES)=={'conjunction_mechanism','conjunction_robustness/reference',
                                       'conjunction_robustness/cluster','conjunction_robustness/estimation'}
    assert set(extension.JSON_TARGETS)=={'conjunction_mechanism/update_structures.json',
                                        'conjunction_robustness/reference/frame_diagnostics.json',
                                        'conjunction_robustness/cluster/summary.json',
                                        'conjunction_robustness/estimation/interval_regression.json'}


def test_verifies_every_table_and_row_ignoring_runtime_metadata(replay_fixture):
    root,expected,output,plain,compressed=replay_fixture
    # Different gzip headers and metadata do not change scientific targets.
    relative=extension.DIRECTORIES[0]+'/curves.csv.gz'
    assert (output/relative).read_bytes()!=(expected/relative).read_bytes()
    receipt=extension.verify(output)
    assert receipt['passed'] and receipt['scientific_rows']==20
    assert len(receipt['csv_checks'])==8 and len(receipt['json_scientific_leaves'])==4
    assert {r['scope'] for r in receipt['csv_checks'].values()}=={'all rows'}
    assert all(r['fields']==3*r['rows'] for r in receipt['csv_checks'].values())
    assert json.loads((output/'extension_verification.json').read_text())==receipt
    assert not (expected/'extension_verification.json').exists()


@pytest.mark.parametrize('row_number',[0,1,2])
def test_a_changed_scientific_value_in_any_row_fails(replay_fixture,row_number):
    _,_,output,plain,_=replay_fixture
    changed=[dict(r) for r in plain];changed[row_number]['estimate']+=.05
    write_table(output/extension.DIRECTORIES[0]/'counts.csv',changed)
    with pytest.raises(AssertionError,match='estimate'):
        extension.verify(output)
    assert not (output/'extension_verification.json').exists()


def test_compressed_table_values_and_row_identities_are_scientific(replay_fixture):
    _,_,output,_,compressed=replay_fixture
    changed=[dict(r) for r in compressed];changed[-1]['dataset']='wrong-frame'
    write_table(output/extension.DIRECTORIES[-1]/'curves.csv.gz',changed)
    with pytest.raises(AssertionError,match='dataset'):
        extension.verify(output)


@pytest.mark.parametrize('kind',['missing_table','extra_table','missing_row','extra_row','changed_columns'])
def test_scientific_inventory_and_shape_fail_closed(replay_fixture,kind):
    _,_,output,plain,_=replay_fixture
    folder=output/extension.DIRECTORIES[1]
    target=folder/'counts.csv'
    if kind=='missing_table':target.unlink()
    elif kind=='extra_table':write_table(folder/'unexpected.csv',plain)
    elif kind=='missing_row':write_table(target,plain[:-1])
    elif kind=='extra_row':write_table(target,plain+[dict(dataset='extra',seed=4,estimate=.4)])
    else:target.write_text('dataset,seed,wrong_name\ncell-A,1,0.1\n')
    with pytest.raises(AssertionError):extension.verify(output)


def test_missing_expected_targets_and_missing_output_folder_fail(replay_fixture):
    _,expected,output,_,_=replay_fixture
    folder=extension.DIRECTORIES[2]
    expected_folder=expected/folder
    expected_folder.rename(expected_folder.with_name('temporarily_absent'))
    with pytest.raises(FileNotFoundError,match='Missing extension targets'):
        extension.verify(output)
    expected_folder.with_name('temporarily_absent').rename(expected_folder)
    (output/folder).rename((output/folder).with_name('temporarily_absent'))
    with pytest.raises(FileNotFoundError):extension.verify(output)


@pytest.mark.parametrize('relative',extension.JSON_TARGETS)
def test_each_scientific_json_is_required_and_compared(replay_fixture,relative):
    _,_,output,_,_=replay_fixture
    target=output/relative
    original=json.loads(target.read_text())
    target.unlink()
    with pytest.raises(FileNotFoundError):extension.verify(output)
    target.write_text(json.dumps({**original,'nested':{'seed':31,'units':207}}))
    with pytest.raises(AssertionError,match='units'):
        extension.verify(output)


def test_finite_tolerance_does_not_allow_material_or_nonfinite_changes(replay_fixture):
    _,_,output,plain,_=replay_fixture
    changed=[dict(r) for r in plain];changed[-1]['estimate']+=1e-10
    target=output/extension.DIRECTORIES[0]/'counts.csv'
    write_table(target,changed)
    assert extension.verify(output)['passed']
    changed[-1]['estimate']='NaN'
    write_table(target,changed)
    with pytest.raises(AssertionError):extension.verify(output)


@pytest.mark.parametrize('relative',['.','artifacts','artifacts/expected','artifacts/expected/sub',
                                    'src/new','data/cache','artifacts/figures','artifacts/validation'])
def test_verification_guard_precedes_read_or_write(replay_fixture,relative):
    root,expected,_,_,_=replay_fixture
    before={p:p.read_bytes() for p in expected.rglob('*') if p.is_file()}
    with pytest.raises(ValueError):extension.verify(root/relative)
    assert all(p.read_bytes()==b for p,b in before.items())


def test_verification_rejects_symlink_alias_and_output_children(replay_fixture):
    root,expected,output,_,_=replay_fixture
    alias=root/'artifacts/alias';alias.symlink_to(expected,target_is_directory=True)
    with pytest.raises(ValueError):extension.verify(alias)
    (output/'hidden-symlink').symlink_to(expected,target_is_directory=True)
    with pytest.raises(ValueError):extension.verify(output)


def test_reproduce_refuses_to_replace_a_prior_run(replay_fixture):
    _,_,output,_,_=replay_fixture
    marker=output/'keep.txt';marker.write_text('preserve this run')
    with pytest.raises(ValueError,match='new or empty'):
        extension.reproduce(output)
    assert marker.read_text()=='preserve this run'


def test_verify_only_uses_the_same_package_relative_output_path(replay_fixture,monkeypatch,tmp_path):
    root,_,output,_,_=replay_fixture
    (output/'extension_run_manifest.json').write_text(json.dumps({'configuration':extension.CONFIG,'passed':True,
                                                               'complete_extension_reproduction':True}))
    outside=tmp_path/'other-cwd';outside.mkdir();monkeypatch.chdir(outside)
    monkeypatch.setattr('sys.argv',['reproduce_extensions','--verify-only','--output',str(output.relative_to(root))])
    extension.main()
    assert (output/'extension_verification.json').exists()


def test_verify_only_requires_matching_successful_configuration(replay_fixture,monkeypatch):
    _,_,output,_,_=replay_fixture
    changed=dict(extension.CONFIG);changed['cluster_replicates']=2
    receipt=output/'extension_run_manifest.json'
    monkeypatch.setattr('sys.argv',['reproduce_extensions','--verify-only','--output',str(output)])
    for payload in [{'configuration':changed,'passed':True},{'configuration':extension.CONFIG,'passed':False}]:
        receipt.write_text(json.dumps(payload))
        with pytest.raises(AssertionError,match='successful matching'):
            extension.main()
