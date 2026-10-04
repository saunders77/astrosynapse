#!/usr/bin/env python3
"""Build differential fixtures from the original project (development only).
Run with the repository's NumPy environment. No Python is needed for hosting.
"""
import argparse,json,sys,random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'astrosynapse2/backend'))
import numpy as np
from astro2.engine import Game,GameConfig,RNGStreams,Action,ActionKind,DecisionFamily
from astro2.baselines import HeuristicChooser
from astro2.cards import CARD_BY_ID
from astro2.engine_encoding import EngineEncoder
from astro2.model import NumpyActor
from astro2.lethal import find_lethal_plan, _possible
class Rng:
    def __init__(self,seed): self.state=seed%2147483646+1
    def next(self): self.state=(self.state*48271)%2147483647; return self.state
    def shuffle(self,a):
        for i in range(len(a)-1,0,-1):
            j=self.next()%(i+1); a[i],a[j]=a[j],a[i]
    def getstate(self): return random.Random(self.state).getstate()
    def setstate(self,state): self.state=state

def observation(o):
    d=o.to_dict()
    for k,v in d.items():
        if isinstance(v,list):
            if k.endswith('_in_play'):
                d[k]=[{**i,'card':i['card']['card_id']} for i in v]
            else:
                d[k]=[i['card_id'] if isinstance(i,dict) and 'card_id' in i else i for i in v]
    return d

def win_probability(actor, encoded):
    if actor.spec.objective_version >= 2:
        logits = actor.predict_values(encoded.state, np.asarray([int(encoded.family)]))[0]
    else:
        options = actor.predict_options(encoded.state, encoded.actions, int(encoded.family))
        logits = options[options.mean(axis=1).argmax()]
    return float(np.mean(1 / (1 + np.exp(-np.clip(logits, -40, 40)))))

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',default='/tmp/astro-reference.json'); p.add_argument('--games',type=int,default=12); a=p.parse_args()
    games=[]; samples={}; kinds=set(); families=set(); played=set()
    for seed in range(730,730+a.games):
        decisions=[]; balanced=HeuristicChooser('balanced'); rng=Rng(seed+908)
        def chooser(pid,d):
            if rng.next()%7==0:
                options=[i for i,v in enumerate(d.actions) if v.kind!=ActionKind.END_TURN]
                if options: return options[rng.next()%len(options)]
            return balanced(pid,d)
        def hook(pid,d,chosen):
            decisions.append({'player':pid,'family':d.family.value,'observation':observation(d.observation),'actions':[v.to_dict() for v in d.actions],'selected':d.actions.index(chosen)})
            if d.family==DecisionFamily.MAIN and _possible(d) and sum('lethal' in v for v in decisions)<2:
                decisions[-1]['lethal']=[[family.value,list(key)] for family,key in find_lethal_plan(g,d)]
            kinds.add(chosen.kind.value); families.add(d.family.value)
            if chosen.kind==ActionKind.PLAY_CARD: played.add(chosen.card_id)
            if d.family.value not in samples and len(d.actions)>1: samples[d.family.value]=d
        g=Game(config=GameConfig(seed=seed,starting_player=seed%2,rules_version=2,max_turns=240,max_actions_per_turn=220),rng_streams=RNGStreams(Rng(seed),Rng(seed+101),Rng(seed+202),Rng(seed+303)),choosers=(chooser,chooser),decision_hook=hook)
        g.run(); games.append({'seed':seed,'starts':seed%2,'decisions':decisions,'result':{'winner':g.result.winner,'turns':g.result.turns,'truncated':g.result.truncated}})
    registry=json.loads((ROOT/'php-game/models/registry.json').read_text())
    actor=NumpyActor.load(ROOT/registry[-1]['source']); encoder=EngineEncoder(version=actor.spec.encoder_version)
    neural=[]
    for d in samples.values():
        e=encoder.encode_decision(d.observation,d)
        neural.append({'family':d.family.value,'observation':observation(d.observation),'actions':[v.to_dict() for v in d.actions], 'state':e.state.tolist(),'encoded_actions':e.actions.tolist(),'scores':actor.predict_options(e.state,e.actions,int(e.family)).mean(axis=1).tolist()})
    levels=[]
    registry=json.loads((ROOT/'php-game/models/registry.json').read_text())
    for entry in registry:
        level_actor=NumpyActor.load(ROOT/entry['source'])
        level_encoder=EngineEncoder(version=level_actor.spec.encoder_version)
        checks=[]
        for d in samples.values():
            e=level_encoder.encode_decision(d.observation,d)
            checks.append({'family':d.family.value,'state':e.state.tolist(),'encoded_actions':e.actions.tolist(),'scores':level_actor.predict_options(e.state,e.actions,int(e.family)).mean(axis=1).tolist(),'win_probability':win_probability(level_actor,e)})
        levels.append({'id':entry['id'],'checks':checks})
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps({'games':games,'neural':neural,'levels':levels},separators=(',',':')))
    print(f'{len(games)} games, {sum(len(g["decisions"]) for g in games)} decisions; {len(played)} played card types, {len(families)} families, {len(kinds)} action kinds. Fixture: {a.output}')
    print('Missing played cards:',[c.name for i,c in CARD_BY_ID.items() if i not in played])
if __name__=='__main__': main()
