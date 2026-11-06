"""Create a new reviewed namespace; leave unused v1 source and receipts intact."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OLD = HERE.with_name('C_native_final_publication_runbook_v1')


def replace_once(value, old, new):
    if value.count(old) != 1:
        raise ValueError('Expected one source fragment: ' + old)
    return value.replace(old, new, 1)


def main():
    source = OLD / 'review_metadata.py'
    if hashlib.sha256(source.read_bytes()).hexdigest() != '95e43930aca1b1f1e544b97863694047b8a1972864180e01ef3849d0a1158b73':
        raise ValueError('Preserved initial reviewer differs')
    text = source.read_text()
    helpers = '''def bind_frozen_row_sources(meta, item, identity, frozen, snapshot, source_root=ROOT):
    """Fresh small-file SHA crossbinding; checkpoint payload is never read."""
    source_root = Path(source_root).resolve()
    def canonical(path):
        path = Path(path)
        return (path if path.is_absolute() else source_root / path).resolve()
    require(identity.get('kind') == 'final' and item['run_id'] == identity.get('run_id') == identity.get('actual_run_id') == frozen['id'], 'Frozen final actual-run identity differs')
    require(identity.get('model_tag') == frozen['model'], 'Frozen identity model differs')
    require(canonical(item['source_checkpoint']) == canonical(identity['checkpoint']) == canonical(frozen['checkpoint_path']), 'Checkpoint canonical path differs from frozen final identity')
    for name in ('config_path', 'summary_path'):
        require(canonical(identity[name]) == canonical(frozen[name]), 'Frozen final source canonical path differs: ' + name)
    try:
        relative = str(canonical(frozen['config_path']).relative_to(source_root))
    except ValueError as error:
        raise ValueError('Frozen config is outside the source workspace') from error
    require(relative in snapshot['configs'], 'Frozen snapshot does not cover config')
    for name, digest_name in [('config_path', 'config_sha256'), ('summary_path', 'summary_sha256')]:
        path = canonical(frozen[name])
        matches = [value for key, value in identity['input_sha256'].items() if canonical(key) == path]
        require(len(matches) == 1 and re.fullmatch('[0-9a-f]{64}', matches[0]) and item[digest_name] == matches[0], 'Row source SHA differs from frozen READY: ' + name)
        if name == 'config_path':
            require(snapshot['configs'][relative]['sha256'] == matches[0], 'Config SHA differs between snapshot and frozen READY')
        raw = meta.opaque(frozen[name], matches[0])  # Fresh bytes, no summary JSON/score parsing.
        if name == 'config_path':
            require(len(raw) == snapshot['configs'][relative]['bytes'], 'Config byte length differs from snapshot')
    observed = meta.payload_stat(item['source_checkpoint'], item['source_stat'])
    require(type(identity.get('checkpoint_bytes')) is int and type(identity.get('checkpoint_mtime_ns')) is int and
            observed['bytes'] == identity['checkpoint_bytes'] and observed['mtime_ns'] == identity['checkpoint_mtime_ns'],
            'Checkpoint size/mtime differs from frozen final identity')


def validate_roundtrip_report(item, base_sha256):
    """Prevent mixing individually valid reports from distinct draft sources."""
    encoded, decoded, verified = (item[name] for name in ('encode', 'decode', 'bytewise_verification'))
    fp = item['model_fingerprint']
    require(re.fullmatch('[0-9a-f]{64}', fp.get('sha256_model_state', '')) and type(fp.get('state_elements')) is int and fp['state_elements'] > 0, 'Complete row model fingerprint required')
    require(encoded.get('status') == 'encoded' and decoded.get('status') == 'decoded_and_hash_verified' and verified.get('status') == 'passed', 'All roundtrip stages must be complete')
    require(all(stage.get('model_fingerprint') == fp for stage in (encoded, decoded, verified)), 'Roundtrip stage fingerprints differ from row')
    require(encoded.get('source_checkpoint_sha256') == verified.get('source_checkpoint_sha256') == item['source_checkpoint_sha256'], 'Nested source checkpoint SHA differs')
    require(encoded.get('archive_sha256') == verified.get('archive_sha256') == item['archive_sha256'], 'Nested ZIP SHA differs')
    require(Path(encoded['archive']).resolve() == Path(decoded['archive']).resolve() == Path(item['archive_path']).resolve(), 'Nested ZIP canonical path differs')
    require(encoded.get('archive_bytes') == item['archive_stat']['bytes'] and encoded.get('source_checkpoint_bytes_including_optimizer') == item['source_stat']['bytes'], 'Nested ZIP/checkpoint byte lengths differ')
    require(encoded.get('base_sha256') == base_sha256, 'Nested encode base SHA differs from frozen native base')
    count = decoded.get('tensor_raw_hashes_verified')
    require(type(count) is int and count > 0 and type(verified.get('tensor_count')) is int and count == verified['tensor_count'], 'Complete decoded tensor hash inventory must match bytewise verification')
    require(encoded.get('model_only') is True and encoded.get('lossless_weight_bits') is True and decoded.get('model_only_evaluation_format') is True and verified.get('all_tensor_bytes_equal') is True, 'Complete lossless model-only roundtrip flags required')


'''
    text = replace_once(text, 'def review(model):', helpers + 'def review(model):')
    text = replace_once(text,
        "        identity_row = finals[item['run_id']]; frozen = frozen_rows[item['run_id']]; definition = definitions[item['run_id']]",
        "        identity_row = finals[item['run_id']]; frozen = frozen_rows[item['run_id']]; definition = definitions[item['run_id']]\n        bind_frozen_row_sources(meta, item, identity_row, frozen, snapshot)")
    text = replace_once(text,
        "            require(item['additional_encoding_calls'] == 1 and item['bytewise_verification'].get('all_tensor_bytes_equal') is True, 'Every unique weight needs complete actual bytewise roundtrip')",
        "            require(item['additional_encoding_calls'] == 1 and item['bytewise_verification'].get('all_tensor_bytes_equal') is True, 'Every unique weight needs complete actual bytewise roundtrip')\n            validate_roundtrip_report(item, manifest['base_dependency']['base_sha256'])")
    with (HERE / 'review_metadata.py').open('x') as f:
        f.write(text)
    with (HERE / 'test_review_metadata.py').open('x') as f:
        f.write((OLD / 'test_review_metadata.py').read_text())
    old_doc = ROOT / 'docs/C1B与C2p8B原final权重包持久发布runbook.md'
    if hashlib.sha256(old_doc.read_bytes()).hexdigest() != '2c554067748a3460879459914768e703f9253f65533283eaa3fdaea90ee37eea':
        raise ValueError('Preserved initial runbook differs')
    doc = old_doc.read_text().replace('C_native_final_publication_runbook_v1/review_metadata.py', 'C_native_final_publication_runbook_v2/review_metadata.py')
    doc = replace_once(doc, 'source checkpoint 身份与 stat。optimizer 不包含',
        'config SHA 同时对冻结 snapshot、READY input SHA 和当前小文件 fresh SHA，summary SHA 对冻结 READY 与当前小文件 fresh SHA；checkpoint 规范路径同时对原 registry/冻结 final identity，实际 size/mtime 对冻结 identity，另核 row 自身 source stat。summary 只散列字节不解析评分语义，checkpoint 不读 payload。独立 unique draft 的 row FP、CP/ZIP/base SHA、ZIP路径/大小、decode 全 tensor hash 状态/数量与 encode/decode/bytewise 报告必须一致，拒绝拼接不同报告。optimizer 不包含')
    doc = replace_once(doc, "  docs/C1B_final_weights_durable_publication.json <<'PY'", "  docs/C1B_final_weights_durable_publication.json \\\n  docs/C1B_final_weights_durable_publication.progress.json <<'PY'")
    doc = replace_once(doc, "r = json.load(open(sys.argv[1])); receipt = sys.argv[2]", "r = json.load(open(sys.argv[1])); receipt = sys.argv[2]; progress = sys.argv[3]")
    doc = replace_once(doc, "'--destination', r['destination'], '--receipt', receipt]", "'--destination', r['destination'], '--receipt', receipt, '--progress', progress]")
    doc += '\n本 v2 保留原 v1/95e 初稿及其准备收据，只修复未实际使用的独立 reviewer 和未来命令；原 v1 publisher 示例缺必需 `--progress`，不应用于执行。ROOT 对两模型均应使用各自独立 progress 路径。此次未运行实际整包 reviewer 或 publisher。\n'
    with (ROOT / 'docs/C1B与C2p8B原final权重包持久发布runbook_v2.md').open('x') as f:
        f.write(doc)


if __name__ == '__main__':
    main()
