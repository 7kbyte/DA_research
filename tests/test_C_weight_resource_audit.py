"""Small receipt/stat reviews never rehash a multi-GB delivery payload."""
import importlib.util,json,stat
from pathlib import Path
import pytest

SPEC=importlib.util.spec_from_file_location('resource_review',Path(__file__).parents[1]/'scripts/audit_C_weight_resource_records.py')
A=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(A)


def publication(tmp_path,monkeypatch):
    destination=tmp_path/'tiny.tar';destination.write_bytes(b'tiny');st=destination.stat()
    details=dict(device=st.st_dev,inode=st.st_ino,bytes=st.st_size,uid=st.st_uid,gid=st.st_gid,
        mode=oct(stat.S_IMODE(st.st_mode)),mtime_ns=st.st_mtime_ns,ctime_ns=st.st_ctime_ns)
    receipt=dict(status='passed',destination=str(destination),source_sha256=A.A_SHA,destination_sha256=A.A_SHA,
        bytes=4,destination_stat=details,file_fsync_after_copy=True,prefix_sha_verified_before_truncate=True,
        file_fsync_after_truncate=True,parent_directory_fsync=True,original_inode_preserved=True,permissions_and_ownership_preserved=True)
    path=tmp_path/'publication.json';path.write_text(json.dumps(receipt));monkeypatch.setattr(A,'A_RECEIPT',path);monkeypatch.setattr(A,'A_BYTES',4)
    return path,destination,receipt


def test_receipt_and_current_stat_review_does_not_hash_tar(tmp_path,monkeypatch):
    receipt,destination,_=publication(tmp_path,monkeypatch);real=A.file_sha256
    def limited(path):
        assert Path(path)!=destination,'TAR unexpectedly rehashed'
        return real(path)
    monkeypatch.setattr(A,'file_sha256',limited)
    reviewed=A.audit_A_publication()
    assert reviewed['status'].startswith('passed') and reviewed['fresh_full_tar_hash_performed_by_this_reviewer'] is False


@pytest.mark.parametrize('mutation',['size','owner','fsync','sha'])
def test_inconsistent_publication_receipt_or_stat_is_rejected(tmp_path,monkeypatch,mutation):
    path,destination,receipt=publication(tmp_path,monkeypatch)
    if mutation=='size':destination.write_bytes(b'changed size')
    elif mutation=='owner':receipt['destination_stat']['uid']+=1
    elif mutation=='fsync':receipt['file_fsync_after_copy']=False
    else:receipt['destination_sha256']='wrong'
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):A.audit_A_publication()
