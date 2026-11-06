"""One-off authorized section7 wording reduction; no render in this script."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT=Path(__file__).resolve().parents[3];BASE=Path(__file__).resolve().parent
source=ROOT/'研究报告.md';before=source.read_text()
assert hashlib.sha256(before.encode()).hexdigest()=='b140726776171970e223c858a6cd6fa1937748bb835898080853b69179527762'
begin=before.index('## 7. 复现与交付状态');end=before.index('## 8. 主要参考文献与资源');section=before[begin:end]
changes=[
('补充A全部126个final逻辑端点现已有持久权重：18个复用主归档，15个来自已发布的验证选λ端点归档，余下93个已补齐，三者共同覆盖完整网格，不重复计算复用端点。此次93份新增权重归档为106,627,983,360字节，自带native初始模型的六文件闭包；每份权重均完成无损往返校验，持久文件经完整SHA写入和读回核验。验证选λ的18个政策端点是其中的子集（15份新增及3个主归档引用），指所选λ下的训练最终模型，不是按验证时间点选择的best。',
 '补充A全部126个final逻辑端点已有持久权重：主归档复用18个、验证选λ归档新增15个及此次补齐93个，共同覆盖完整网格。93份归档为106,627,983,360字节，自带native六文件基座，完成无损往返及持久SHA写入和读回核验。验证选λ的18个端点是其中子集（15份新增、3个主归档引用），为所选λ的训练最终模型，不是验证时间点选择的best。'),
('补充B全部60个final端点也已持久发布，共60份独立权重，归档69,578,864,640字节，自带native六文件基座与恢复工具。另对事前固定的B4、训练/生成seed11、draw1001、K20最终模型，在全新目录仅用该持久包实际完成CPU恢复：',
 '补充B全部60个final端点已持久发布，含60份独立权重，归档69,578,864,640字节，自带native六文件基座和工具。对事前固定的B4、训练/生成seed11、draw1001、K20最终模型，在全新目录仅用该包实际完成CPU恢复：'),
('每份新权重均完成逐张量字节往返校验，持久文件经完整SHA读回验证；恢复还需要主归档中的初始模型基座，C410 final归档单独不构成完整闭包。',
 '权重经逐张量字节往返、持久文件经完整SHA读回核验；恢复仍需主归档中的初始模型基座，C410 final归档单独不构成完整闭包。'),
('经过无损往返和完整发布SHA核验。此best归档自带native六文件基座，不需要从主归档另取基座，与上述final归档的依赖不同。',
 '经无损往返和完整发布SHA核验，且自带native六文件基座，恢复不再依赖主归档基座。'),
('best权重保留各自验证选择的实际step与目标/回放曝光，不作为新增训练run。1B、2.8B和行为评价按冻结完整合同继续执行，尚未完整验收的产物不提前列为已交付。恢复依赖和空间要求见',
 'best保留各自验证选择的实际step与目标/回放曝光，不算新增训练run。1B、2.8B与行为评价仍按冻结完整合同执行，未完整验收的产物不列为已交付。恢复依赖及容量见'),
('共享存储出现 ENOSPC 后，工作树迁至本机大盘，原路径通过链接保留。',
 '共享存储ENOSPC后，工作树迁至本机大盘，原路径由链接保留。'),
('另在全新目录中，仅从两份持久TAR实际恢复了C410的B4 seed11最终模型，完整指纹匹配，整次检查耗时39.2秒。这验证该固定单例的两包恢复闭包，不等于重新训练或全部24端点再次独立恢复。',
 '另在全新目录仅从两份持久TAR实际恢复C410的B4 seed11最终模型，完整指纹匹配，耗时39.2秒。这只验证该单例的两包恢复闭包，不代表重新训练或全部24端点独立恢复。')]
for old,new in changes:assert section.count(old)==1;section=section.replace(old,new)
removed=end-begin-len(section);assert 100<=removed<=150,removed
after=before[:begin]+section+before[end:]
assert after[:begin]==before[:begin] and after[after.index('## 8. 主要参考文献与资源'):]==before[end:]
# Keep every link and numeral token in the delivery section, except the unchanged
# AI authorship prose has no scientific numerical implication. Replacements
# themselves above allow the human to review exact before/after wording.
import re
assert re.findall(r'\]\(([^)]*)\)',section)==re.findall(r'\]\(([^)]*)\)',before[begin:end])
assert re.findall(r'\d[\d,]*(?:\.\d+)?',section)==re.findall(r'\d[\d,]*(?:\.\d+)?',before[begin:end])
retained=BASE/'retained_before_layout';retained.mkdir()
proofs=[]
for name in ['研究报告.md','reports/研究报告_工作稿.pdf','reports/研究报告_工作稿.html','reports/研究报告_工作稿.render.json']:
    p=ROOT/name;dest=retained/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    proofs.append({'path':name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'copy':str(dest.relative_to(ROOT))})
(BASE/'retained_manifest.json').write_text(json.dumps(proofs,ensure_ascii=False,indent=2)+'\n')
source.write_text(after);(BASE/'applied_section7.md').write_text(section)
value={'status':'applied_authorized_section7_wording_reduction','old_MD_sha256':hashlib.sha256(before.encode()).hexdigest(),'new_MD_sha256':hashlib.sha256(after.encode()).hexdigest(),'characters_removed':removed,'scientific_prefix_bytes_identical':True,'references_suffix_bytes_identical':True,'prefix_sha256':hashlib.sha256(before[:begin].encode()).hexdigest(),'all_section7_numerals_and_links_retained_in_original_order':True,'font_figures_and_renderer_unchanged':True,'exact_wording_changes':[{'before':a,'after':b} for a,b in changes]}
(BASE/'application_receipt.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in value.items() if k!='exact_wording_changes'}))
