import json, glob, os, math
ART = sorted(glob.glob('results/microbench/costm-decode-*.json'), key=os.path.getmtime)[-1]
d = json.load(open(ART))

# --- estrazione: tempi TOTALI di splitk-idot per shape, e minimo fra varianti ---
idot, best = {}, {}
for c in d['cells']:
    if c['kernel'] != 'gemm-kquant-multirow' or not c.get('M'): continue
    sh = c['shape']; k = (c['variant'].split('/')[0], sh.get('K'), sh.get('N')); M = c['M']
    us = c['msPerOp']['p50'] * 1000
    if 'splitk-idot' in c['variant']: idot.setdefault(k, {})[M] = us
    cur = best.setdefault(k, {}).get(M)
    if cur is None or us < cur: best[k][M] = us

SHAPES = [(('q4_K',2048,512), 'q4_K 2048x512 · expert gate/up', '#c0392b'),
          (('q4_K',512,2048), 'q4_K 512x2048 · expert down',    '#2980b9'),
          (('q6_K',512,2048), 'q6_K 512x2048 · expert down',    '#27ae60'),
          (('q8_0',2048,4096),'q8_0 2048x4096 · attn q-proj',   '#8e44ad')]

def fit(pts):
    n=len(pts); sx=sum(p[0] for p in pts); sy=sum(p[1] for p in pts)
    sxx=sum(p[0]**2 for p in pts); sxy=sum(p[0]*p[1] for p in pts)
    b=(n*sxy-sx*sy)/(n*sxx-sx*sx); return (sy-b*sx)/n, b

def svg(w,h,body,title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'font-family="ui-sans-serif,system-ui,sans-serif" font-size="12">'
            f'<title>{title}</title><rect width="{w}" height="{h}" fill="#fff"/>{body}</svg>')

def axes(x0,y0,x1,y1,xlab,ylab):
    s = f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="#333"/>'
    s += f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#333"/>'
    s += f'<text x="{(x0+x1)/2}" y="{y1+42}" text-anchor="middle" fill="#333">{xlab}</text>'
    s += f'<text x="{x0-46}" y="{(y0+y1)/2}" text-anchor="middle" fill="#333" transform="rotate(-90 {x0-46} {(y0+y1)/2})">{ylab}</text>'
    return s

# ============ GRAFICO 1: T(M) totale e il fit affine ============
X0,Y0,X1,Y1 = 78,34,700,336
MS=[1,2,4,8,16]
lx=lambda M:(X0+((M-1)/15)*(X1-X0))   # ASSE LINEARE: e' l'unico su cui una funzione affine SI VEDE come retta
body = axes(X0,Y0,X1,Y1,'M (righe per dispatch) — scala LINEARE','tempo TOTALE del dispatch (µs)')
ymax=26
for M in MS: body+=f'<line x1="{lx(M):.1f}" y1="{Y0}" x2="{lx(M):.1f}" y2="{Y1}" stroke="#eee"/><text x="{lx(M):.1f}" y="{Y1+18}" text-anchor="middle" fill="#555">{M}</text>'
for v in range(0,ymax+1,5):
    yy=Y1-(v/ymax)*(Y1-Y0); body+=f'<line x1="{X0}" y1="{yy:.1f}" x2="{X1}" y2="{yy:.1f}" stroke="#eee"/><text x="{X0-8}" y="{yy+4:.1f}" text-anchor="end" fill="#555">{v}</text>'
for i,(k,lab,col) in enumerate(SHAPES[:3]):
    r=idot.get(k,{}); pts=[(M,r[M]) for M in MS if M in r]
    if len(pts)<3: continue
    a,b=fit(pts)
    y=lambda v:Y1-(v/ymax)*(Y1-Y0)
    body+=f'<line x1="{lx(1):.1f}" y1="{y(a+b*1):.1f}" x2="{lx(16):.1f}" y2="{y(a+b*16):.1f}" stroke="{col}" stroke-width="1" stroke-dasharray="5 4" opacity=".55"/>'
    body+='<polyline fill="none" stroke="'+col+'" stroke-width="2.2" points="'+' '.join(f'{lx(M):.1f},{y(v):.1f}' for M,v in pts)+'"/>'
    for M,v in pts: body+=f'<circle cx="{lx(M):.1f}" cy="{y(v):.1f}" r="3.4" fill="{col}"/>'
    body+=f'<text x="{X1+8}" y="{y(a+b*16)+4:.1f}" fill="{col}" font-size="11">{lab.split(" · ")[0]}</text>'
    body+=f'<text x="{X0+14}" y="{Y0+16+i*17}" fill="{col}" font-size="11">T(M) = {a:.2f} + {b:.3f}·M  µs</text>'
body+=f'<text x="{X0}" y="{Y0-12}" fill="#111" font-weight="600" font-size="13">Il tempo totale è AFFINE in M — la retta tratteggiata è il fit</text>'
open('docs/deep-dive/img/costm-totale.svg','w').write(svg(820,392,body,'T(M) totale e fit affine'))

# ============ GRAFICO 2: medio vs marginale ============
X1b=640
body = axes(X0,Y0,X1b,Y1,'M (righe per dispatch) — scala log2','costo per riga (µs)')
lx=lambda M:(X0+(math.log2(M)/math.log2(16))*(X1b-X0))
ymax2=8
for M in MS: body+=f'<line x1="{lx(M):.1f}" y1="{Y0}" x2="{lx(M):.1f}" y2="{Y1}" stroke="#eee"/><text x="{lx(M):.1f}" y="{Y1+18}" text-anchor="middle" fill="#555">{M}</text>'
for v in range(0,ymax2+1,2):
    yy=Y1-(v/ymax2)*(Y1-Y0); body+=f'<line x1="{X0}" y1="{yy:.1f}" x2="{X1b}" y2="{yy:.1f}" stroke="#eee"/><text x="{X0-8}" y="{yy+4:.1f}" text-anchor="end" fill="#555">{v}</text>'
y=lambda v:Y1-(v/ymax2)*(Y1-Y0)
k,lab,col=SHAPES[0]; r=idot.get(k,{}); pts=[(M,r[M]) for M in MS if M in r]; a,b=fit(pts)
body+='<polyline fill="none" stroke="#c0392b" stroke-width="2.4" points="'+' '.join(f'{lx(M):.1f},{y(v/M):.1f}' for M,v in pts)+'"/>'
for M,v in pts:
    body+=f'<circle cx="{lx(M):.1f}" cy="{y(v/M):.1f}" r="3.6" fill="#c0392b"/>'
    body+=f'<text x="{lx(M)+(14 if M==1 else 0):.1f}" y="{y(v/M)+(14 if M==1 else -10):.1f}" text-anchor="middle" fill="#c0392b" font-size="10">{v/M:.2f}</text>'
body+=f'<line x1="{lx(1):.1f}" y1="{y(b):.1f}" x2="{lx(16):.1f}" y2="{y(b):.1f}" stroke="#111" stroke-width="2" stroke-dasharray="7 4"/>'
body+=f'<text x="{lx(16)-6:.1f}" y="{y(b)-9:.1f}" text-anchor="end" fill="#111" font-size="11">costo MARGINALE b = {b:.3f} µs/riga — PIATTO</text>'
body+=f'<text x="{X0+170}" y="{Y0+80}" fill="#c0392b" font-size="11">costo MEDIO = b + a/M — cala del 26% a ogni raddoppio PER SEMPRE</text>'
body+=f'<text x="{X0+170}" y="{Y0+97}" fill="#555" font-size="11">…anche quando il marginale è già a zero: è ammortamento dell&#39;intercetta</text>'
body+=f'<text x="{X0}" y="{Y0-12}" fill="#111" font-weight="600" font-size="13">Perché «non ha saturato» era un artefatto — q4_K 2048x512</text>'
open('docs/deep-dive/img/costm-medio-vs-marginale.svg','w').write(svg(760,392,body,'medio vs marginale'))

# ============ GRAFICO 3: la banda tradisce la L2 ============
pts=[]
for c in d['cells']:
    w=c.get('weightBytesPerToken'); t=c['msPerOp']['p50']
    if w and t and c.get('M'): pts.append(w*c['M']/(t/1000)/1e9)
pts=sorted(pts,reverse=True)[:26]
X1c=620; body=axes(X0,Y0,X1c,Y1,'celle del banco, ordinate per banda','banda effettiva sui pesi (GB/s)')
ymax3=850
for v in range(0,ymax3+1,200):
    yy=Y1-(v/ymax3)*(Y1-Y0); body+=f'<line x1="{X0}" y1="{yy:.1f}" x2="{X1c}" y2="{yy:.1f}" stroke="#eee"/><text x="{X0-8}" y="{yy+4:.1f}" text-anchor="end" fill="#555">{v}</text>'
bw=(X1c-X0)/len(pts)
for i,g in enumerate(pts):
    hh=(g/ymax3)*(Y1-Y0); col='#c0392b' if g>435 else '#95a5a6'
    body+=f'<rect x="{X0+i*bw+1:.1f}" y="{Y1-hh:.1f}" width="{bw-2:.1f}" height="{hh:.1f}" fill="{col}"/>'
yv=Y1-(435/ymax3)*(Y1-Y0)
body+=f'<line x1="{X0}" y1="{yv:.1f}" x2="{X1c}" y2="{yv:.1f}" stroke="#111" stroke-width="2"/>'
body+=f'<text x="{X1c-4}" y="{yv-8:.1f}" text-anchor="end" fill="#111" font-size="11">tetto VRAM MISURATO della scheda: 435 GB/s</text>'
body+=f'<text x="{X0+10}" y="{Y0+18}" fill="#c0392b" font-size="11">{sum(1 for g in pts if g>435)} celle SOPRA il tetto fisico — fino a {pts[0]:.0f} GB/s</text>'
body+=f'<text x="{X0+10}" y="{Y0+35}" fill="#555" font-size="11">impossibile dalla VRAM: i pesi stanno nella L2 (64 MB), le matrici sono 0,6-17,8 MB</text>'
body+=f'<text x="{X0}" y="{Y0-12}" fill="#111" font-weight="600" font-size="13">Il banco NON misura il regime del decode</text>'
open('docs/deep-dive/img/costm-banda-l2.svg','w').write(svg(700,392,body,'banda effettiva vs tetto VRAM'))

# ============ GRAFICO 4: righe per expert ============
X1d=620; body=axes(X0,Y0,X1d,Y1,'M (token nella finestra)','righe medie viste da un expert attivo')
MM=[1,2,4,8,16,32,64,128,256]
lx=lambda M:(X0+(math.log2(M)/math.log2(256))*(X1d-X0))
ymax4=9
for M in MM: body+=f'<text x="{lx(M):.1f}" y="{Y1+18}" text-anchor="middle" fill="#555">{M}</text>'
for v in range(0,ymax4+1,1):
    yy=Y1-(v/ymax4)*(Y1-Y0); body+=f'<line x1="{X0}" y1="{yy:.1f}" x2="{X1d}" y2="{yy:.1f}" stroke="#eee"/><text x="{X0-8}" y="{yy+4:.1f}" text-anchor="end" fill="#555">{v}</text>'
y=lambda v:Y1-(v/ymax4)*(Y1-Y0)
ind=[(M, 8*M/(256*(1-(248/256)**M))) for M in MM]
cor=[(M, 8*M/(8+1.4*(M-1))) for M in MM]
body+='<polyline fill="none" stroke="#95a5a6" stroke-width="2.2" stroke-dasharray="6 4" points="'+' '.join(f'{lx(M):.1f},{y(min(v,ymax4)):.1f}' for M,v in ind)+'"/>'
body+='<polyline fill="none" stroke="#27ae60" stroke-width="2.4" points="'+' '.join(f'{lx(M):.1f},{y(min(v,ymax4)):.1f}' for M,v in cor)+'"/>'
for M,v in cor:
    if M in (2,4,16,64,256): body+=f'<circle cx="{lx(M):.1f}" cy="{y(min(v,ymax4)):.1f}" r="3.4" fill="#27ae60"/><text x="{lx(M):.1f}" y="{y(min(v,ymax4))-10:.1f}" text-anchor="middle" fill="#27ae60" font-size="10">{v:.1f}</text>'
body+=f'<line x1="{X0}" y1="{y(8):.1f}" x2="{X1d}" y2="{y(8):.1f}" stroke="#111" stroke-dasharray="3 3"/>'
body+=f'<text x="{X1d-4}" y="{y(8)-7:.1f}" text-anchor="end" fill="#111" font-size="11">tetto teorico: 8 righe (top-8)</text>'
body+=f'<text x="{X0+12}" y="{Y0+18}" fill="#27ae60" font-size="11">routing CORRELATO (stimato da recall 82,67%) — DA VERIFICARE con lo spike (2)</text>'
body+=f'<text x="{X0+12}" y="{Y0+35}" fill="#95a5a6" font-size="11">routing INDIPENDENTE (limite inferiore teorico)</text>'
body+=f'<text x="{X0}" y="{Y0-12}" fill="#111" font-weight="600" font-size="13">Perché il segmento expert non può usare M grande — top-8 su 256</text>'
open('docs/deep-dive/img/costm-righe-per-expert.svg','w').write(svg(700,392,body,'righe per expert'))
print('4 SVG scritti in docs/deep-dive/img/')
