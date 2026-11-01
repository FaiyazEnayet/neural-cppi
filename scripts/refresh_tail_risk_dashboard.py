"""Refresh presentation from committed dashboard evidence, without retraining.

Run make_tail_risk_charts.py first. The original dashboard generator also
attaches the same historical summary through portfolio_ui/renderer.py.
"""
from pathlib import Path
import base64,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'portfolio_ui'))
from renderer import page_parts,attach_tail_risk

def main():
 path=ROOT/'docs/dashboard.html'
 raw=path.read_text(encoding='utf8')
 evidence=json.JSONDecoder().raw_decode(raw.split('const EVIDENCE=',1)[1])[0]
 original_figures=json.dumps(evidence['figures'],sort_keys=True)
 original_tables=json.dumps(evidence['tables'],sort_keys=True)
 attach_tail_risk(evidence,ROOT)
 head,js=page_parts(evidence)
 library=(ROOT/'portfolio_ui/plotly.min.js').read_text(encoding='utf8')
 payload=json.dumps(evidence,ensure_ascii=False,separators=(',',':')).replace('</','<'+chr(92)+'/')
 output=head+'<script>'+library+'</script><script>const EVIDENCE='+payload+';'+chr(10)+js+'</script></body></html>'
 assert json.dumps(evidence['figures'],sort_keys=True)==original_figures
 assert json.dumps(evidence['tables'],sort_keys=True)==original_tables
 path.write_text(output,encoding='utf8')
 print('Refreshed historical study presentation; existing research figures and tables preserved exactly.')

if __name__=='__main__':main()
