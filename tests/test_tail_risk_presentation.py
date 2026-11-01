"""Public aggregate presentation contracts; no access to the private report."""
from pathlib import Path
import base64,json,re,struct

ROOT=Path(__file__).resolve().parents[1]

def summary():
 return json.loads((ROOT/'docs/tail-risk-summary.json').read_text())

def test_tier_scopes_and_ablation_are_separate():
 d=summary();t=d['tiers']
 assert t['test']+t['held_out']+t['international']==t['sealed']
 assert t['validation']+t['layer_tuning']==t['tuning']
 assert t['sealed']+t['tuning']==d['windows']
 assert d['ablation']['windows']!=d['windows']
 assert 'Great Depression' in d['ablation']['subset']
 assert t['sealed_main_windows']+t['international']==t['sealed']

def test_aggregate_distribution_keeps_losses():
 d=summary()
 assert len(d['differences'])==d['main_windows']
 assert sum(v>0 for v in d['differences'])==d['terminal_wins']
 assert float(d['recovery']['difference']) in d['differences']
 assert d['shipped_breaches']==d['classic_breaches']==0

def test_publication_charts_are_embedded_without_external_dependencies():
 raw=(ROOT/'docs/dashboard.html').read_text(encoding='utf8')
 e=json.JSONDecoder().raw_decode(raw.split('const EVIDENCE=',1)[1])[0]
 assert e['tail_risk']==summary()
 for key in ['breaches','terminal']:
  content=(ROOT/('docs/tail-risk-'+key+'.png')).read_bytes()
  assert content[:8]==b'\x89PNG\r\n\x1a\n'
  assert struct.unpack('>I',content[16:20])[0]==1200
  assert base64.b64decode(e['tail_risk_images'][key].split(',',1)[1])==content
 assert not re.search(r'<script[^>]+src=',raw,re.I)
 assert 'riskReadout(wealthResult,wf);historicalTailRisk(g);' in raw
 assert 'function neural(){historicalTailRisk();' not in raw
 assert raw.index('href="#wealth"') < raw.index('href="#tail-risk"')

def test_readme_section_is_complete_and_uses_published_paths():
 section=(ROOT/'historical-tail-risk-section.md').read_text(encoding='utf8')
 readme=(ROOT/'README.md').read_text(encoding='utf8')
 assert section.strip() in readme
 assert section.startswith('## Historical tail-risk challenge')
 assert chr(0x2014) not in section
 assert summary()['report_sha256'] in section
 for key in ['breaches','terminal']:assert 'docs/tail-risk-'+key+'.png' in section
 assert '25 sealed windows' in section and '13 tuning-tier windows' in section
 assert 'Classic CPPI also recorded zero breaches.' in section
