#!/usr/bin/env python3
"""Read-only, version-bound PDF/HTML QA; writes only an independent review namespace."""
from __future__ import annotations
import datetime
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import unicodedata
from urllib.parse import unquote, urlsplit
import pypdfium2 as pdfium
from PIL import Image, ImageDraw

ROOT = Path('/data/202610/7kb/research').resolve()
OUT = Path(__file__).resolve().parent

def sha(data): return hashlib.sha256(data).hexdigest()
def proof(path):
    p = Path(path)
    return {'path': str(p.resolve()), 'sha256': sha(p.read_bytes()), 'bytes': p.stat().st_size}
def stable_bytes(path):
    p = Path(path)
    def stat():
        s = p.stat()
        return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]
    before = stat(); data = p.read_bytes(); after = stat()
    assert before == after, str(p) + ' changed while reading'
    return data, before
def new_json(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2); f.write('\n')
def norm(text):
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', text)).replace('\u200b', '').replace('\u00ad', '')

class Parser(HTMLParser):
    def __init__(self):
        super().__init__(); self.links=[]; self.images=[]; self.ids=set(); self.blocks=[]; self.block=None; self.styles=[]; self.in_style=False
    def handle_starttag(self, tag, attrs):
        a=dict(attrs)
        if 'id' in a: self.ids.add(a['id'])
        if tag=='a' and 'href' in a: self.links.append(a['href'])
        if tag=='img' and 'src' in a: self.images.append(a['src'])
        if tag in ('p','h1','h2','h3','h4','td','th'):
            self.block={'tag':tag,'text':''}
        if tag=='style': self.in_style=True
    def handle_endtag(self, tag):
        if self.block is not None and tag==self.block['tag']:
            self.blocks.append(self.block); self.block=None
        if tag=='style': self.in_style=False
    def handle_data(self, data):
        if self.block is not None: self.block['text']+=data
        if self.in_style: self.styles.append(data)

def main():
    timer=ROOT/'artifacts/provenance/report_delivery_section7_update_v3/render_whole_child_timing.json'
    td=json.loads(timer.read_text()); assert type(td['returncode']) is int and td['returncode']==0
    source=ROOT/'研究报告.md'; html=ROOT/'reports/研究报告_工作稿.html'; pdf=ROOT/'reports/研究报告_工作稿.pdf'; meta=pdf.with_suffix('.render.json')
    data={}; stats={}
    for name,path in [('source',source),('html',html),('pdf',pdf),('render',meta)]:
        data[name],stats[name]=stable_bytes(path)
    md=json.loads(data['render']); assert sha(data['source'])==md['source_sha256']; assert sha(data['pdf'])==md['pdf_sha256']
    font=Path(md['font']); assert sha(font.read_bytes())==md['font_sha256']
    assert 'C410_nll_independent_review_v1' in data['source'].decode() and 'C410两包权重实际恢复验收.md' in data['source'].decode()
    for key,ext in [('source','md'),('html','html'),('pdf','pdf'),('render','json')]:
        with (OUT/f'reviewed_{key}_snapshot.{ext}').open('xb') as f:f.write(data[key])
    parser=Parser(); parser.feed(data['html'].decode())
    cssurls=re.findall(r'url\(\s*[\'"]?([^\)\'\"]+)', '\n'.join(parser.styles))
    links=[]; missing=[]; resources=[]
    for kind,values in [('href',parser.links),('img',parser.images),('font_or_CSS_url',cssurls)]:
        for value in values:
            parts=urlsplit(value)
            if parts.scheme or parts.netloc:
                links.append({'kind':kind,'url':value,'external_not_fetched':True}); continue
            if not parts.path:
                exists=not parts.fragment or unquote(parts.fragment) in parser.ids
                item={'kind':kind,'url':value,'local_anchor':True,'exists':exists}
            else:
                target=(html.parent/unquote(parts.path)).resolve()
                item={'kind':kind,'url':value,'resolved':str(target),'exists':target.exists()}
                if target.exists() and kind!='href':resources.append(proof(target))
            links.append(item)
            if not item['exists']:missing.append(item)
    assert not missing, missing
    doc=pdfium.PdfDocument(data['pdf']); n=len(doc); pages=[]; full=[]; images=[]
    for i,page in enumerate(doc):
        textpage=page.get_textpage(); text=textpage.get_text_range(); full.append(text)
        png=OUT/f'page_{i+1:02}.png'; image=page.render(scale=1.45).to_pil().convert('RGB'); image.save(png); images.append(png)
        gray=image.convert('L'); hist=gray.histogram(); ink=sum(hist[:240])/(image.width*image.height)
        pages.append({'page':i+1,'width_pt':page.get_width(),'height_pt':page.get_height(),'text_characters':len(text),'nonwhite_fraction_below_240':ink,'footer_page_number_present':norm(f'{i+1} / {n}') in norm(text),'C410_section_or_recovery_text':any(s in text for s in ('410M完整结果','40,029,696','逐张量恢复','C410','两包')),'page_image':proof(png)})
        textpage.close(); page.close()
    doc.close()
    fulltext='\n'.join(full)
    with (OUT/'extracted_all_pages.txt').open('x') as f:f.write(fulltext)
    for start in range(0,n,4):
        subset=images[start:start+4]; tilew=600; tileh=875
        sheet=Image.new('RGB',(tilew*2,tileh*2),'#d7dce1'); draw=ImageDraw.Draw(sheet)
        for offset,p in enumerate(subset):
            im=Image.open(p);im.thumbnail((tilew-16,tileh-32));x=(offset%2)*tilew+(tilew-im.width)//2;y=(offset//2)*tileh+25
            sheet.paste(im,(x,y));draw.text(((offset%2)*tilew+12,(offset//2)*tileh+6),f'Page {start+offset+1} / {n}',fill='black')
        sheet.save(OUT/f'contact_{start+1:02}_{min(start+4,n):02}.png')
    pdfnorm=re.sub(r'固定回放预算下的领域适应与遗忘\d+/'+str(n), '', norm(fulltext)); uncovered=[]; checked=0
    for block in parser.blocks:
        text=norm(block['text'])
        if not text:continue
        checked+=1
        if text not in pdfnorm:
            uncovered.append({'tag':block['tag'],'text':block['text']})
    keywords=['6.3长程与较大训练设置','40,029,696','35,025,984','55,040,832','50,214.2','3109.7','106,627,983,360','69,578,864,640','26,490,163,200','B固定首例实际恢复验收','全部126个final逻辑端点']
    keycheck={s:norm(s) in pdfnorm for s in keywords}
    receipt={'status':'prepared_for_human_visual_review','reviewed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'render_child_timer':proof(timer),'render_child_rc0':True,'source_current_SHA_matches_render':True,'PDF_SHA_matches_render':True,'source_stat5_at_read':stats,'inputs':{k:{'path':str(p.resolve()),'sha256':sha(data[k]),'bytes':len(data[k])} for k,p in [('source',source),('html',html),('pdf',pdf),('render',meta)]},'renderer_source':proof(ROOT/'scripts/render_report.py'),'font':proof(font),'PDFium_version':pdfium.PYPDFIUM_INFO.version,'pages':pages,'page_count':n,'relative_resources_and_anchors':links,'missing_local_resources':missing,'embedded_resource_proofs':resources,'HTML_text_blocks_checked':checked,'HTML_text_blocks_not_exactly_found_in_PDF':uncovered,'C410_keyword_checks':keycheck,'blank_page_candidates':[p['page'] for p in pages if p['nonwhite_fraction_below_240']<.001],'all_page_number_footers_found':all(p['footer_page_number_present'] for p in pages),'visual_review_not_yet_claimed':True,'no_training_scoring_or_source_modification':True}
    new_json(OUT/'automated_QA.json',receipt)
    print(json.dumps({'pages':n,'missing_local_resources':len(missing),'relative_links':sum('exists' in r for r in links),'text_blocks':checked,'nonexact_blocks':len(uncovered),'C410_keywords':keycheck,'C410_pages':[p['page'] for p in pages if p['C410_section_or_recovery_text']],'output':str(OUT)},ensure_ascii=False))

if __name__=='__main__':main()
