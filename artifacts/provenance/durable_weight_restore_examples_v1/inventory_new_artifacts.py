from pathlib import Path
import hashlib,json
root=Path.cwd();records=[]
for path in sorted(root.rglob('*')):
 if not path.is_file() or path.name in {'artifact_inventory.json','actual_restore_receipt.json'}:continue
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 records.append({'path':str(path.relative_to(root)),'bytes':path.stat().st_size,'sha256':h.hexdigest()})
json.dump({'status':'passed','files':records,'scope':'All completed new QA files incl selected base/ZIPs/new restored models, scripts, manifests, sidecar and closed logs; excludes inventory/receipt self references; no source TAR or original CP hashed'},open('artifact_inventory.json','w'),indent=2)
