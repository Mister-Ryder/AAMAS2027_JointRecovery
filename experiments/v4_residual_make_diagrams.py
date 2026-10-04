"""Editable residual-method schematic; mathematical toy, no measured results."""
from pathlib import Path
import argparse
import subprocess

from experiments.v4_make_diagrams import Diagram,INK,BLUE,ORANGE,TEAL,PURPLE,GRAY


def method(out):
    d=Diagram(1040,388)
    for x,title in ((0,'(a) Shared executed prefix'),(354,'(b) Joint residual prediction'),
                    (708,'(c) Budgeted improvement')):
        d.cell('',x,0,332,280,'rounded=0;strokeColor=#CBD2D9;fillColor=#FFFFFF;')
        d.text(title,x+12,8,310,26,18,bold=True)
    a=d.box('Commitments C + deliberate releases E<br>fixed base B; compatible pool R',12,43,308,46,BLUE,'#F0F7FB')
    d.text('Example response interactions',16,94,299,20,14,GRAY)
    ids={}
    for key,label,x,y,col in (('u','u:8',54,124,BLUE),('v','v:8',226,124,ORANGE),
                            ('z','z:6',54,184,BLUE),('w','w:6',226,184,ORANGE)):
        ids[key]=d.cell(label,x,y,54,30,'ellipse;strokeWidth=%s;strokeColor=%s;fillColor=#FFFFFF;align=center;'
            %(3 if key=='u' else 1.6,col))
    for one,two in (('u','v'),('u','z'),('v','w'),('u','w'),('v','z')):
        d.edge(ids[one],ids[two],GRAY,False,straight=True)
    d.text('Executed warm: L = 8',16,219,302,20,15,TEAL,True)
    d.text('Clique bound: U = max{8,8,6} + 6 = 14',16,242,302,21,14,GRAY)
    enc=d.box('Shared encoder for observable (R, W)<br>rewards · conflicts · resource factors',368,43,304,46,TEAL,'#EFF8F4')
    occ=d.box('Sparse factor readout<br>z → p, with Σᵥ∈Q pᵥ ≤ 1',368,106,304,43,TEAL)
    head=d.box('Request head<br>native workpoint + remaining time',368,165,304,43,TEAL)
    value=d.box('r̂ = L + (U − L) · sigmoid(f)<br>ĝ = q + r̂',368,225,304,43,TEAL,'#EFF8F4')
    d.edge(enc,occ,TEAL);d.edge(occ,head,TEAL);d.edge(head,value,TEAL)
    d.edge(ids['w'],enc,BLUE,points=((343,199),(343,65)))
    choose=d.box('Prioritize unspent requests<br>[ĝ − best gain]₊ / expected cost',722,43,304,46,PURPLE,'#F5F2FA')
    native=d.box('Execute finite native search<br>use the same actual W for every policy',722,106,304,43,PURPLE)
    check=d.box('Check original edges + objective<br>accept only timely improvements',722,165,304,43,PURPLE)
    result=d.box('Report shared prefix and extra gain<br>strict return time includes full validation',722,225,304,43,PURPLE,'#F5F2FA')
    d.edge(choose,native,PURPLE);d.edge(native,check,PURPLE);d.edge(check,result,PURPLE)
    d.edge(value,choose,PURPLE,points=((697,247),(697,65)))
    # A separate offline strip makes supervision/deployment separation visible.
    d.cell('',0,292,1040,62,'rounded=0;strokeColor=#CBD2D9;fillColor=#F8F9FA;')
    d.text('Offline only',12,302,110,23,15,GRAY,True)
    state=d.box('Same actual state<br>and common warm W',130,301,200,43,GRAY)
    alternatives=d.box('Clone → execute alternatives<br>full-budget value / returned mask',354,301,287,43,GRAY)
    train=d.box('Executed-value ranking + mask supervision<br>train the joint residual model',665,301,362,43,GRAY)
    d.edge(state,alternatives,GRAY);d.edge(alternatives,train,GRAY)
    d.text('All methods pay the same prefix; Greedy-only stops there. The 4-node example is mathematical, not an experimental outcome.',
        8,361,1024,23,14,GRAY)
    d.save(out/'method_joint_residual.drawio')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',default='paper/v4_figures/residual_schematics')
    p.add_argument('--export',action='store_true');args=p.parse_args()
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True);method(out)
    if args.export:
        startup=subprocess.STARTUPINFO();startup.dwFlags|=subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow=subprocess.SW_HIDE
        source=out/'method_joint_residual.drawio'
        for kind in ('png','svg','pdf'):
            subprocess.run(['C:/Program Files/draw.io/draw.io.exe','-x','-f',kind]+
                ([] if kind=='png' else ['-e'])+['-b','4','-o',str(source.with_suffix('.'+kind)),str(source)],
                check=True,timeout=60,startupinfo=startup)
    print(str(out))


if __name__=='__main__':main()
