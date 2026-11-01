"""Render aggregate publication charts from the supplied verified study summary.

No model fitting, policy selection, or stress evaluation is performed here.
"""
from pathlib import Path
import base64,io,json,re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator
from fontTools.ttLib import TTFont
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
D=json.loads((ROOT/'docs/tail-risk-summary.json').read_text())
BG='#101214';SURFACE='#1a1b1e';GOLD='#c6a66b';TEXT='#e8e6df';SLATE='#8b949e';LINE='#343638'

def register(name):
 fonts=json.loads((ROOT/'portfolio_ui/fonts.json').read_text())
 raw=base64.b64decode(re.findall(r'base64,([^\)]+)',fonts[name])[-1])
 font=TTFont(io.BytesIO(raw));font.flavor=None
 folder=ROOT/'.cache/tail-fonts';folder.mkdir(parents=True,exist_ok=True)
 path=folder/(name.replace(' ','')+'.ttf');font.save(path)
 font_manager.fontManager.addfont(path)
 return font_manager.FontProperties(fname=path).get_name()

DISPLAY=register('Cormorant Garamond');MONO=register('IBM Plex Mono')
plt.rcParams.update({'figure.facecolor':BG,'axes.facecolor':BG,'text.color':TEXT,
 'axes.labelcolor':TEXT,'xtick.color':SLATE,'ytick.color':SLATE,'axes.edgecolor':LINE,
 'font.family':MONO,'font.size':10,'savefig.facecolor':BG})

def heading(fig,title,subtitle):
 fig.text(.075,.945,'NEURAL-CPPI / HISTORICAL TAIL-RISK CHALLENGE',color=GOLD,fontsize=10)
 fig.text(.075,.885,title,fontfamily=DISPLAY,fontsize=31)
 fig.text(.075,.835,subtitle,color=SLATE,fontsize=10)
 fig.add_artist(plt.Line2D([.075,.94],[.805,.805],transform=fig.transFigure,color=LINE,lw=1))

def foot(fig,main=False):
 scope='Main-set aggregate: 18 sealed + 13 candidate-tuning windows.' if main else 'Aggregate scope: 25 sealed + 13 candidate-tuning windows.'
 fig.text(.075,.070,scope+' Shipped system trained and selected on none.',color=SLATE,fontsize=8)
 fig.text(.075,.045,'Source: verified private archive summary. Overlapping windows are not independent trials.',color=SLATE,fontsize=8)
 fig.text(.075,.022,'Report SHA-256: '+D['report_sha256'],color=SLATE,fontsize=7)

def breaches():
 fig=plt.figure(figsize=(12,7.2),dpi=100)
 heading(fig,'Anytime floor breaches','Daily-close wealth below 90; initial wealth 100. Both complete systems recorded zero breaches.')
 ax=fig.add_axes([.34,.24,.51,.49]);ys=[2,1,0]
 for y,total,breaches,color in zip(ys,[D['windows'],D['windows'],D['ablation']['windows']],[D['shipped_breaches'],D['classic_breaches'],D['ablation']['breaches']],[GOLD,SLATE,GOLD]):
  ax.barh(y,total,height=.28,facecolor='none',edgecolor=LINE,linewidth=1.3)
  if breaches:ax.barh(y,breaches,height=.28,color=color)
  else:ax.scatter([0],[y],s=70,facecolor=BG,edgecolor=color,linewidth=2,zorder=4,clip_on=False)
  ax.text(40,y,str(breaches)+' of '+str(total),va='center',color=color,fontsize=17)
 ax.set_xlim(0,38);ax.set_ylim(-.55,2.5);ax.set_yticks([])
 ax.set_xticks([0,11,38]);ax.set_xlabel('Windows · outlined track shows each configuration’s evaluation set',labelpad=14,fontsize=9)
 ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='x',length=0,pad=9)
 labels=[('Shipped system','Learned policy + deterministic safety layer'),('Classic CPPI','Same multiplier 4.0 and transaction costs'),('Learned policy alone','No safety layer · Great Depression subset')]
 for y,(title,sub) in zip(ys,labels):
  yy=fig.transFigure.inverted().transform(ax.transData.transform((0,y)))[1]
  fig.text(.075,yy+.013,title,fontfamily=DISPLAY,fontsize=22)
  fig.text(.075,yy-.025,sub,color=SLATE,fontsize=8)
 fig.text(.075,.115,'Ablation: 11 Great Depression windows only. Its denominator differs from the full 38-window evaluation.',fontsize=9,color=TEXT)
 foot(fig);fig.savefig(ROOT/'docs/tail-risk-breaches.png',dpi=100);plt.close(fig)

def terminal():
 fig=plt.figure(figsize=(12,8.4),dpi=100)
 heading(fig,'Terminal wealth: the full paired distribution','31 US windows · shipped system minus classic CPPI · wealth starts at 100')
 vals=D['differences'];median=float(D['median_advantage']);outlier=float(D['recovery']['difference'])
 ax=fig.add_axes([.09,.46,.84,.30]);detail=fig.add_axes([.09,.17,.84,.18])
 for axes,bins in [(ax,np.arange(-30,11,1)),(detail,np.arange(-3,9.25,.25))]:
  axes.hist(vals,bins=bins,color=GOLD,edgecolor=BG,linewidth=.7)
  axes.axvline(0,color=SLATE,lw=.8,alpha=.7)
  axes.axvline(median,color=TEXT,ls='--',lw=1.4)
  axes.spines[['top','right']].set_visible(False);axes.grid(axis='y',color=LINE,alpha=.5);axes.set_axisbelow(True)
  axes.yaxis.set_major_locator(MaxNLocator(integer=True));axes.set_ylabel('Windows',fontsize=9)
 ax.set_xlim(-30,10);ax.set_xticks([-30,-20,-10,0,10]);ax.set_ylim(bottom=0)
 ax.annotate(D['recovery']['difference']+'\n1932 trough start',xy=(outlier,1),xytext=(-24,ax.get_ylim()[1]*.48),color=TEXT,fontsize=10,arrowprops={'arrowstyle':'-','color':SLATE},linespacing=1.6)
 ax.annotate('Median '+D['median_advantage'],xy=(median,ax.get_ylim()[1]*.82),xytext=(3,ax.get_ylim()[1]*.84),fontsize=10,color=TEXT,arrowprops={'arrowstyle':'-','color':TEXT})
 detail.set_xlim(-3,9);detail.set_title('Central range enlarged; the -27.24 observation remains visible in the full distribution above.',loc='left',fontsize=9,color=SLATE,pad=12)
 detail.set_xlabel('Paired terminal wealth difference · wealth units',labelpad=10)
 foot(fig,main=True);fig.savefig(ROOT/'docs/tail-risk-terminal.png',dpi=100);plt.close(fig)

if __name__=='__main__':
 assert len(D['differences'])==31
 breaches();terminal()
 print('Wrote both 1200-pixel publication charts from the declared aggregate summary.')
