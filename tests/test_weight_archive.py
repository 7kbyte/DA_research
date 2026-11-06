"""Raw-bit roundtrip including NaN payloads, signed zero and wrong-base rejection."""
import json
import zipfile

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from replay_study.weight_archive import encode_weights, decode_weights, verify_roundtrip


def fixture(tmp_path):
    patterns = np.array([0, 0x80000000, 0x7f800000, 0xff800000, 0x7fc00001, 0x7fa12345,
                         0xffc0beef, 1, 0x80000001, 0x007fffff, 0x00800000, 0x3f800000], dtype=np.uint32)
    state = {"z.special": torch.from_numpy(patterns.view(np.float32)).reshape(3,4),
             "a.scalar": torch.tensor(-0., dtype=torch.float32), "m.empty": torch.empty((0,2))}
    base = tmp_path/"base.safetensors"
    save_file({k: torch.zeros_like(v, dtype=torch.float16) for k,v in state.items()}, str(base))
    checkpoint = tmp_path/"latest.pt"
    torch.save({"model_state":state,"optimizer_state":{"irrelevant":torch.ones(100)},
                "config":{"model_revision":"fixed"},"counters":{"target_prediction_tokens":100},
                "provenance":{"test_scores_accessed":False}},checkpoint)
    return checkpoint,base


def test_special_bit_patterns_multichunk_and_model_metadata_roundtrip(tmp_path):
    source, base = fixture(tmp_path)
    archive, restored = tmp_path/"model.xor.zip", tmp_path/"restored.pt"
    report = encode_weights(source,base,archive,chunk_bytes=16,threads=1)
    decoded = decode_weights(archive,base,restored,threads=1)
    verified = verify_roundtrip(source,restored,archive,threads=1)
    assert verified["all_tensor_bytes_equal"] and verified["tensor_count"]==3
    assert report["model_fingerprint"]==decoded["model_fingerprint"]==verified["model_fingerprint"]
    out = torch.load(restored,weights_only=False)
    assert "optimizer_state" not in out and out["counters"]["target_prediction_tokens"]==100
    assert out["config"]["model_revision"]=="fixed" and decoded["model_only_evaluation_format"]
    with zipfile.ZipFile(archive) as z:
        metadata=json.loads(z.read("metadata.json"))
    special=next(r for r in metadata["tensors"] if r["name"]=="z.special")
    assert len(special["chunks"])==3 and all(r["raw_sha256"] for r in metadata["tensors"])


def test_changed_base_is_rejected_before_any_output_and_source_is_untouched(tmp_path):
    source, base = fixture(tmp_path)
    original=source.read_bytes()
    archive=tmp_path/"model.zip"
    encode_weights(source,base,archive,threads=1)
    wrong=tmp_path/"wrong.safetensors"
    save_file({"wrong":torch.ones(3)},str(wrong))
    output=tmp_path/"must_not_exist"/"restored.pt"
    with pytest.raises(ValueError,match="base SHA256"):
        decode_weights(archive,wrong,output,threads=1)
    assert not output.parent.exists() and source.read_bytes()==original


def test_audit_fingerprint_mismatch_cannot_commit_archive(tmp_path):
    source,base=fixture(tmp_path)
    output=tmp_path/"bad.zip"
    with pytest.raises(ValueError,match="audited model"):
        encode_weights(source,base,output,expected_model_sha256="bad",threads=1)
    assert not output.exists() and not output.with_name(output.name+".partial").exists()


def test_best_checkpoint_kind_and_selection_survive_as_source_metadata(tmp_path):
    source,base=fixture(tmp_path)
    saved=torch.load(source,weights_only=False)
    saved.pop("optimizer_state")
    saved.update(checkpoint_kind="model_only_best_target_validation",
                 selection={"criterion":"target_validation_nll","step":3,"target_prediction_tokens":40})
    torch.save(saved,source)
    archive,restored=tmp_path/"best.zip",tmp_path/"best.pt"
    encode_weights(source,base,archive,threads=1)
    decode_weights(archive,base,restored,threads=1)
    out=torch.load(restored,weights_only=False)
    assert out["checkpoint_kind"]=="model_only_lossless_xor_restored"
    assert out["lossless_delivery_provenance"]["source_checkpoint_kind"]==saved["checkpoint_kind"]
    assert out["selection"]==saved["selection"]
    assert verify_roundtrip(source,restored,archive,threads=1)["all_tensor_bytes_equal"]


@pytest.mark.parametrize("damage",["truncated_zip","changed_payload"])
def test_damaged_archive_is_rejected_without_committing_output(tmp_path,damage):
    source,base=fixture(tmp_path)
    good,bad=tmp_path/"good.zip",tmp_path/"bad.zip"
    encode_weights(source,base,good,threads=1)
    if damage=="truncated_zip":
        bad.write_bytes(good.read_bytes()[:-64])
    else:
        with zipfile.ZipFile(good) as src,zipfile.ZipFile(bad,"w",compression=zipfile.ZIP_DEFLATED) as dst:
            changed=False
            for item in src.infolist():
                payload=src.read(item.filename)
                if not changed and item.filename.endswith(".planes"):
                    payload=bytes([payload[0]^1])+payload[1:];changed=True
                dst.writestr(item.filename,payload)
    restored=tmp_path/"must_not_commit"/"model.pt"
    with pytest.raises((ValueError,zipfile.BadZipFile,EOFError)):
        decode_weights(bad,base,restored,threads=1)
    assert not restored.exists() and not restored.with_name(restored.name+".partial").exists()
