# Copyright (c) 2026 Frame Bridge contributors.
# SPDX-License-Identifier: Apache-2.0
"""Train Relay's original tiny causal Transformer on synthetic household-cat episodes.

Development-only: Python, NumPy, CPU PyTorch. No real-cat recordings, external
corpus, pretrained weights, network inference or user data. Labels are an
original design teacher: agreement measures imitation, not animal fidelity.
The packaged app performs bounded inference in C# without Python/PyTorch.

Reproduce: python tools/train_kitty_policy.py --work-dir <scratch-directory>
See docs/RELAY-TRAINING.json for the exact committed training configuration.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
import argparse
import hashlib
import json
import math
from pathlib import Path
import tempfile
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SEED = 20261101
INPUTS, WIDTH, HEADS, FF, CONTEXT = 22, 96, 6, 192, 32
ARCHITECTURE = 'relay-causal-transformer-156k-v1'
TRAINED_PARAMETERS = 155848
NAMES = ('Watch','Walk','Chase','Swat','SmackPanel','Zoomies','Groom','Sleep','Grumpy','Stretch',
         'Sniff','Knead','Loaf','SlowBlink','LookAround','Stalk','Pounce','Rub','Scratch','Inspect',
         'Yawn','Chirp','Meow','Roll','Circle','Backoff','Peek','ButtWiggle','PawLick','FaceWash',
         'EarFlick','TailChase','Flop','Doze','Startle')
ATTENTION = ('Cursor','Environment','Self','Rest','Memory')
INPUT_NAMES = ('Energy','Sleepiness','Irritation','Stimulation','CursorNear','CursorMotion',
 'ClickPressure','Idle','PanelNear','Affection','ActionAge','RoamDrive','Curiosity','Trust',
 'Habituation','Novelty','SocialNeed','Startle','PlayMemory','RestMemory','ContactMemory','Completion')
SCENARIOS = ('quiet_home','prey_play','repeated_cursor','gentle_contact','rapid_pestering',
 'sudden_novelty','gradual_novelty','play_then_groom','settle_nap_wake','surface_patrol',
 'solitary_play','social_choice','recall_play','recall_contact','recall_startle','recall_novelty',
 'mixed_play_social','mixed_novel_rest','spaced_pet_then_space','poke_then_recovery',
 'calm_handling_then_release','cat_led_social_choice','heldout_social_then_novel',
 'heldout_play_then_startle','heldout_handling_then_play','heldout_poke_then_pet')
BASE = np.array([.78,.12,.04,.08,.15,.05,0,.3,.5,.65,.3,.25,.65,.75,.2,.05,.45,.02,.1,.15,.1,.1],dtype=np.float32)


def softmax(x):
    x=x-x.max(axis=-1,keepdims=True)
    x=np.exp(x)
    return x/x.sum(axis=-1,keepdims=True)

def teacher(x, trace):
    e,s,i,a,n,m,c,idle,p,aff,age,roam,cur,trust,habit,nov,social,startle,play,rest,contact,completion = x.T
    # These traces are hidden from the current input: causal attention must
    # learn to retrieve earlier stimulation and novelty from the episode.
    wave = n*m
    past_wave,past_near,past_novel = trace.T
    hunt = np.maximum(wave,past_wave*.75)*(1-habit*.65)
    calm = np.maximum(0,1-a-i)
    scores = np.array([
        1.4+.5*n-.8*s-a-i,
        .1+2.8*roam+.5*e-2*s-i,
        -2+4*hunt+2*a+e-2.5*s-3*i,
        -1+3.1*n+1.1*hunt+.6*a-1.5*i-1.5*habit,
        -1.9+3.3*p+roam+a-2*s-2*i,
        -3.8+8*a+1.8*e+.8*past_wave-4*i-4*s,
        -.1+1.6*idle+aff+1.4*play-1.4*roam-3.8*a-2.4*i,
        -3.1+6.5*s+2.8*(1-e)+idle-5*a-2*c-i,
        -3+7.8*i+.7*c+startle-.3*aff,
        -.9+2.7*rest+1.1*s+.8*idle-2.6*a-i,
        -1.4+3*nov+1.5*past_novel+1.1*cur+.6*p-2*a-2*i,
        -1.8+2.5*rest+2.1*trust+1.4*s+calm-2.8*a-2*i,
        -1.1+2.1*s+1.3*idle+1.3*trust+calm-2.5*a-2*i,
        -1.5+2.8*trust+1.1*aff+.8*past_near+.5*calm-2*a-2*i,
        -.1+2.1*cur+1.1*nov+habit-.5*n-1.7*s-i,
        -1.7+3.9*hunt+1.2*cur+e-a-2*i-2*s,
        -2.4+4.5*past_wave+1.5*a+e+completion-2*i-2.2*s,
        -2+2.4*trust+1.9*contact+.8*p+aff-2.5*a-3*i,
        -2+2.7*p+1.4*roam+1.2*e-1.2*s-1.5*a-2*i,
        -1.5+3*nov+2.3*cur+roam-2*a-2*i,
        -2.6+3.3*rest+2.4*s+1.1*idle-3*a-2*i,
        -2.2+2.4*past_wave+1.7*play+1.1*social+trust-2*i-2*s,
        -2.8+3.4*social+1.8*contact+trust+.4*n-2*i-1.5*s,
        -2.5+2.2*trust+aff+1.6*calm+.6*e-1.5*s-2*i,
        -2+2.8*s+2*rest+.7*roam+calm-3*a-2*i,
        -3+5*i+2*startle-trust,
        -2.4+2.5*startle+1.6*cur+1.1*nov-a,
        -3+3.8*hunt+1.3*e+.5*completion-2*s-i,
        -2+2.5*play+aff+calm-2.5*a-i,
        -2.6+2*play+1.1*idle+aff+calm-2*a-i,
        -2+.9*nov+.6*past_novel+cur+.3*idle-.7*s-i,
        -3+2.5*roam+1.5*e+1.1*habit+1.2*a-2*i-3*s,
        -3+2.2*trust+aff+1.1*calm+1.2*play-1.7*s-2*i,
        -2.3+3.4*s+1.8*idle+trust+.7*rest-2.7*a-i,
        -4+6.2*startle+1.3*nov-1.5*trust,
    ]).T
    attention = np.array([
        -1+3.2*n+2*hunt+social-3*habit-2*i-2*s,
        .5+2.2*cur+2*nov+1.2*habit-1.8*s-.7*i,
        .1+2*i+1.4*play+.8*contact+.5*calm,
        -1.2+4*s+2*(1-e)+1.5*rest-3*a,
        -.2+1.4*rest+1.1*play+1.3*past_novel+habit,
    ]).T
    return softmax(scores/.85),softmax(attention/.85)


def labels(current, memory, stage_action=None):
    """Original soft teacher; memory is privileged training state, never an input."""
    action, gaze = teacher(current, memory[:, :3])
    scores, attention = np.log(action + 1e-12)*.85, np.log(gaze + 1e-12)*.85
    e,s,i,a,n,m,c,idle,p,aff,age,roam,cur,trust,habit,nov,social,startle,play,rest,contact,completion=current.T
    wave,near,novel,pressure,gentle,rested,repetition=memory.T
    after_play=wave*(1-n*m)*(1-i)
    scores[:,6]+=3.5*after_play
    scores[:,28]+=4.8*after_play
    scores[:,29]+=3.3*after_play
    scores[:,25]+=7.0*pressure*(1-trust*.35)
    scores[:,26]+=4.5*pressure*(1-c)
    scores[:,8]+=3.5*pressure
    scores[:,13]+=4.6*gentle*(1-i)
    scores[:,17]+=3.0*gentle*(1-i)
    scores[:,10]+=2.2*novel*(1-nov)
    scores[:,19]+=2.2*novel*(1-nov)
    scores[:,14]+=1.8*repetition
    scores[:,30]+=1.0*repetition
    scores[:,9]+=2.5*rested*(1-s)
    scores[:,20]+=2.0*rested*(1-s)
    attention[:,0]-=2.6*repetition+2.5*pressure
    attention[:,1]+=2.0*repetition
    attention[:,2]+=3.0*after_play+1.4*pressure
    attention[:,3]+=1.3*rested*s
    attention[:,4]+=3.8*novel*(1-nov)+1.5*pressure*(1-c)
    if stage_action is not None:
        active=stage_action>=0
        # Stage plans are varied in length and onset. Physical eligibility and
        # urgent needs still take precedence over a suggested routine step.
        scores[np.arange(len(scores))[active],stage_action[active]]+=.65*(1-i[active])
    return softmax(scores/.85).astype(np.float32),softmax(attention/.85).astype(np.float32)


def update_memory(memory, x):
    decay=np.array([.82,.88,.86,.88,.88,.90,.90],dtype=np.float32)
    near,motion,pressure=x[:,4],x[:,5],x[:,6]
    events=np.column_stack((near*motion,near,x[:,15],pressure,
        x[:,20]*(1-pressure)*x[:,13],x[:,1]*x[:,7],near*(1-motion)))
    return memory*decay+events*(1-decay)


def episodes(seed,count,mixes=False,only_scenarios=None):
    rng=np.random.default_rng(seed)
    kinds=np.array(only_scenarios if only_scenarios is not None else range(26 if mixes else 22))
    scenario=rng.choice(kinds,count)
    lengths=rng.integers(12,CONTEXT+1,count)
    x=np.zeros((count,CONTEXT,INPUTS),np.float32)
    action=np.zeros((count,CONTEXT,35),np.float32)
    gaze=np.zeros((count,CONTEXT,5),np.float32)
    memory=np.zeros((count,7),np.float32)
    baseline=np.clip(BASE+rng.normal(0,.09,(count,INPUTS)),0,1)
    baseline[:,6]=0; baseline[:,17]*=.2
    offsets=rng.uniform(-.06,.06,count)
    recall_duration=rng.uniform(.25,.8,count)
    for t in range(CONTEXT):
        phase=np.clip(t/np.maximum(lengths-1,1)+offsets,0,1)
        current=np.clip(baseline+rng.normal(0,.015,(count,INPUTS)),0,1)
        current[:,10]=(phase*5)%1; current[:,21]=np.maximum(0,(phase*5)%1-.85)*5
        current[:,6]=0; current[:,15]*=.3
        stage=np.full(count,-1,np.int64)
        def sel(k): return scenario==k
        def plan(k,steps):
            mask=sel(k)
            stage[mask]=np.array(steps)[np.minimum((phase[mask]*len(steps)).astype(int),len(steps)-1)]
        # Quiet autonomous activities, including self-directed attention.
        q=sel(0); current[q,4:7]=0; current[q,7]=phase[q]; current[q,1]=.12+phase[q]*.35
        plan(0,[14,13,30,6,28,29,12,33])
        # Predatory play has a pause/aim/capture/inspection arc, not endless chase.
        q=sel(1); hunt=q&(phase>.12)&(phase<.7)
        current[q,0]=.9; current[q,1]=.06; current[q,2]=.03
        current[hunt,4]=.9; current[hunt,5]=.7; current[hunt,3]=.35+phase[hunt]*.45
        plan(1,[0,15,27,16,2,3,19,28])
        # Repetition causes habituation; a late small novelty can renew interest.
        q=sel(2); current[q,4]=.8; current[q,5]=.09; current[q,14]=phase[q]
        current[q,15]=np.where(phase[q]>.86,.65,.01); current[q,12]=.5
        plan(2,[0,3,14,30,12,14,10])
        # Spaced, gentle contact leaves a different history from rapid pestering.
        q=sel(3); current[q,13]=.88; current[q,2]=.01; current[q,3]=.1
        current[q,4]=.55; current[q,6]=np.where((t%7)<2,.15,0)
        current[q,20]=np.where((t%7)<3,.85,.3); current[q,16]=.6
        plan(3,[0,13,17,22,13,23,32])
        q=sel(4); current[q,4]=.95; current[q,6]=np.where(phase[q]<.68,.85,0)
        current[q,2]=np.minimum(.95,phase[q]*1.5); current[q,13]=.65-phase[q]*.35
        current[q,17]=np.where(phase[q]<.68,.35,.1); current[q,3]=.55
        plan(4,[0,8,8,25,25,26,14])
        q=sel(5); current[q,15]=np.where((phase[q]>.2)&(phase[q]<.45),1,.05)
        current[q,17]=np.where((phase[q]>.2)&(phase[q]<.45),.95,.1)
        current[q,13]=.25; current[q,12]=.6; current[q,4]=.1
        plan(5,[0,34,25,26,26,10,14])
        q=sel(6); current[q,15]=.15+phase[q]*.45; current[q,17]=.01
        current[q,13]=.85; current[q,12]=.9; current[q,8]=.85
        plan(6,[14,10,19,10,17])
        q=sel(7); current[q,18]=.85; current[q,3]=.5*(1-phase[q]); current[q,4:7]=0
        current[q,7]=.5; current[q,9]=.75; current[q,13]=.8
        plan(7,[19,28,29,6,28,29,32,12])
        q=sel(8); current[q,4:7]=0; current[q,13]=.9; current[q,19]=phase[q]
        current[q,1]=np.where(phase[q]<.75,.4+phase[q]*.7,.2)
        current[q,0]=np.where(phase[q]<.75,.5-phase[q]*.25,.8)
        plan(8,[24,11,12,33,7,7,9,20])
        q=sel(9); current[q,8]=.95; current[q,11]=.8; current[q,12]=.85
        current[q,15]=.45; current[q,0]=.85; current[q,1]=.1
        plan(9,[1,19,10,4,18,19,1])
        q=sel(10); current[q,0]=.95; current[q,1]=.04; current[q,11]=.9
        current[q,3]=.55; current[q,4:7]=0; current[q,14]=.7
        plan(10,[1,31,5,23,32,9])
        q=sel(11); current[q,4]=.65; current[q,5]=.02; current[q,13]=.9
        current[q,16]=.9; current[q,20]=.5; current[q,2]=.01
        plan(11,[13,22,17,14,12,13,21])
        # Recall cases deliberately converge to the exact same observations.
        # Their labels can only be distinguished using earlier tokens.
        for k in range(12,16):
            q=sel(k); cue=q&(phase<recall_duration); current[q]=BASE
            if k==12: current[cue,4:6]=.95; current[cue,3]=.6
            elif k==13: current[cue,20]=.95; current[cue,13]=.9; current[cue,6]=.12
            elif k==14: current[cue,6]=.95; current[cue,17]=.7; current[cue,13]=.3
            else: current[cue,15]=.95; current[cue,12]=.9
        # Different ordered compound events are excluded from training and
        # evaluated separately, instead of leaking test episodes into training.
        for k in (16,17,22,23,24,25):
            q=sel(k); first=q&(phase<.45); second=q&(phase>=.45)
            if k in (16,23): current[first,4:6]=.9; current[first,3]=.6
            if k==16: current[second,20]=.8; current[second,13]=.9
            if k==17:
                current[first,15]=.9; current[second,1]=.8; current[second,4:7]=0
            if k==22:
                current[first,20]=.8; current[first,13]=.9; current[second,15]=.8
            if k==23:
                current[second,17]=.9; current[second,6]=.7; current[second,13]=.3
            if k==24:
                current[first,20]=.9; current[first,13]=.9; current[first,3]=.07
                current[second,4:6]=.85; current[second,3]=.5
            if k==25:
                current[first,6]=.9; current[first,2]=.65; current[first,13]=.35
                current[second,20]=.8; current[second,13]=.6; current[second,6]=.08
        # Bond/handling cues use the existing contact/trust channels, not an
        # extra sensor or a claim that the model understands human gestures.
        for k in (18,19,20,21):
            q=sel(k); earlier=q&(phase<.6); later=q&(phase>=.6)
            current[q,4:7]=0; current[q,13]=.85; current[q,2]=.02
            if k==18:
                current[earlier,6]=np.where((t%8)<2,.12,0)
                current[earlier,20]=np.where((t%8)<3,.85,.3)
                current[q,16]=.65; current[later,20]=.25
            elif k==19:
                current[earlier,6]=.9; current[earlier,2]=.7; current[earlier,13]=.35
                current[later,2]=.4*(1-phase[later]); current[later,13]=.55
            elif k==20:
                current[earlier,20]=.85; current[earlier,13]=.9
                current[q,3]=.06; current[q,9]=.8; current[later,20]=.3
            else:
                current[q,4]=.7; current[q,5]=.02; current[q,14]=.7
                current[q,16]=.65; current[q,20]=np.where(phase[q]<.3,.6,.1)
                current[q,7]=phase[q]; current[q,1]=.15+phase[q]*.3
        current=np.clip(current,0,1).astype(np.float32)
        memory=update_memory(memory,current)
        a,g=labels(current,memory,stage)
        x[:,t]=current; action[:,t]=a; gaze[:,t]=g
    valid=np.arange(CONTEXT)[None,:]<lengths[:,None]
    return x,action,gaze,valid,scenario


def position_encoding():
    positions=np.arange(CONTEXT,dtype=np.float32)[:,None]
    denominator=np.exp(np.arange(0,WIDTH,2,dtype=np.float32)*(-math.log(10000)/WIDTH))
    result=np.empty((CONTEXT,WIDTH),np.float32)
    result[:,0::2]=np.sin(positions*denominator)
    result[:,1::2]=np.cos(positions*denominator)
    return result


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1=nn.LayerNorm(WIDTH,eps=1e-5)
        self.qkv=nn.Linear(WIDTH,3*WIDTH)
        self.projection=nn.Linear(WIDTH,WIDTH)
        self.norm2=nn.LayerNorm(WIDTH,eps=1e-5)
        self.ff=nn.Linear(WIDTH,FF)
        self.ffout=nn.Linear(FF,WIDTH)
    def forward(self,x):
        b,t,_=x.shape
        q,k,v=self.qkv(self.norm1(x)).chunk(3,dim=-1)
        q,k,v=(a.reshape(b,t,HEADS,WIDTH//HEADS).transpose(1,2) for a in (q,k,v))
        weights=q@k.transpose(-2,-1)/math.sqrt(WIDTH//HEADS)
        mask=torch.ones(t,t,dtype=torch.bool,device=x.device).triu(1)
        weights=weights.masked_fill(mask,float('-inf')).softmax(-1)
        attended=(weights@v).transpose(1,2).contiguous().reshape(b,t,WIDTH)
        x=x+self.projection(attended)
        return x+self.ffout(F.gelu(self.ff(self.norm2(x)),approximate='tanh'))


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.input=nn.Linear(INPUTS,WIDTH)
        self.register_buffer('positions',torch.tensor(position_encoding()))
        self.blocks=nn.ModuleList([Block(),Block()])
        self.finalnorm=nn.LayerNorm(WIDTH,eps=1e-5)
        self.behavior=nn.Linear(WIDTH,35)
        self.attention=nn.Linear(WIDTH,5)
    def forward(self,x):
        x=self.input(x*2-1)+self.positions[:x.shape[1]]
        for block in self.blocks: x=block(x)
        x=self.finalnorm(x)
        return self.behavior(x),self.attention(x)


def metrics(model,data,batch=128):
    x,a,g,valid,scenario=data
    actions=[]; gazes=[]
    with torch.no_grad():
        for start in range(0,len(x),batch):
            pa,pg=model(torch.from_numpy(x[start:start+batch]))
            actions.append(pa.softmax(-1).numpy()); gazes.append(pg.softmax(-1).numpy())
    pa,pg=np.concatenate(actions),np.concatenate(gazes)
    def measure(pred,target):
        delta=np.log(np.maximum(target,1e-9))-np.log(np.maximum(pred,1e-9))
        return {'top1_agreement':float((pred.argmax(-1)==target.argmax(-1))[valid].mean()),
            'mean_kl_nats':float((target*delta).sum(-1)[valid].mean()),
            'mean_total_variation':float((.5*np.abs(pred-target).sum(-1))[valid].mean())}
    result={'episodes':len(x),'valid_timesteps':int(valid.sum()),
        'behavior':measure(pa,a),'attention':measure(pg,g),'by_scenario':{}}
    for index in np.unique(scenario):
        mask=valid&(scenario==index)[:,None]
        result['by_scenario'][SCENARIOS[index]]={'timesteps':int(mask.sum()),
            'behavior_agreement':float((pa.argmax(-1)==a.argmax(-1))[mask].mean()),
            'attention_agreement':float((pg.argmax(-1)==g.argmax(-1))[mask].mean())}
    return result


def probe_sequences():
    result={}
    for name in ('quiet','play','gentle','pester','novel','repeated','gradual'):
        sequence=np.tile(BASE,(20,1)).copy()
        for t in range(12):
            if name=='play': sequence[t,4:6]=.95; sequence[t,3]=.6
            elif name=='gentle': sequence[t,20]=.95; sequence[t,13]=.9; sequence[t,6]=.12
            elif name=='pester': sequence[t,6]=.95; sequence[t,17]=.7; sequence[t,13]=.3
            elif name=='novel': sequence[t,15]=.95; sequence[t,12]=.9
            elif name=='repeated': sequence[t,4]=.9; sequence[t,5]=.01; sequence[t,14]=t/12
            elif name=='gradual': sequence[t,15]=t/12*.7; sequence[t,13]=.9
        result[name]=sequence
    return result


def predictions(model,x):
    with torch.no_grad():
        a,g=model(torch.tensor(np.asarray(x[-CONTEXT:])[None],dtype=torch.float32))
        return a[0,-1].softmax(-1).numpy(),g[0,-1].softmax(-1).numpy()


def sequence_checks(model):
    sequences=probe_sequences(); outputs={name:predictions(model,x) for name,x in sequences.items()}
    result={}
    for left,right in [('play','quiet'),('gentle','pester'),('novel','quiet'),('novel','repeated'),('pester','quiet')]:
        a,g=outputs[left]; b,h=outputs[right]
        assert np.array_equal(sequences[left][-1],sequences[right][-1])
        result[left+'_vs_'+right]={'identical_final_observation':True,
            'behavior_total_variation':float(.5*np.abs(a-b).sum()),
            'attention_total_variation':float(.5*np.abs(g-h).sum()),
            'left_top_behavior':NAMES[a.argmax()],'right_top_behavior':NAMES[b.argmax()],
            'left_top_attention':ATTENTION[g.argmax()],'right_top_attention':ATTENTION[h.argmax()]}
    ablations={}
    for name,x in sequences.items():
        a,g=outputs[name]
        last=np.tile(x[-1],(len(x),1)); perm=x.copy()
        perm[:-1]=perm[np.random.default_rng(SEED+17).permutation(len(x)-1)]
        b,h=predictions(model,last); c,j=predictions(model,perm)
        ablations[name]={'repeat_final_behavior_tv':float(.5*np.abs(a-b).sum()),
            'repeat_final_attention_tv':float(.5*np.abs(g-h).sum()),
            'shuffle_prefix_behavior_tv':float(.5*np.abs(a-c).sum()),
            'shuffle_prefix_attention_tv':float(.5*np.abs(g-j).sum())}
    # Train-time context matters only if matched-current histories measurably
    # change both heads. These are synthetic counterfactuals, not pet cognition.
    assert max(v['behavior_total_variation'] for v in result.values())>.08,result
    assert max(v['attention_total_variation'] for v in result.values())>.04,result
    return {'counterfactuals':result,'history_ablations':ablations}


def export(model,root,report):
    def write_atomic(path,text):
        # Readers see either the previous complete artifact or the next one.
        # In particular, a UI build must never parse a half-written weight file.
        with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',newline='\n',
                prefix='.'+path.name+'.',suffix='.pending',dir=path.parent,delete=False) as temporary:
            temporary.write(text)
            temporary_path=Path(temporary.name)
        os.replace(temporary_path,path)
    arrays=[]
    def add(name,value): arrays.append((name,value.detach().cpu().numpy().astype(np.float32)))
    def linear(prefix,layer): add(prefix+'Weights',layer.weight.T.contiguous()); add(prefix+'Bias',layer.bias)
    linear('Input',model.input)
    add('PositionEncoding',model.positions)
    for index,block in enumerate(model.blocks):
        prefix='Block'+str(index)
        add(prefix+'Norm1Scale',block.norm1.weight); add(prefix+'Norm1Bias',block.norm1.bias)
        linear(prefix+'Qkv',block.qkv); linear(prefix+'Projection',block.projection)
        add(prefix+'Norm2Scale',block.norm2.weight); add(prefix+'Norm2Bias',block.norm2.bias)
        linear(prefix+'FeedForward',block.ff); linear(prefix+'FeedForwardOutput',block.ffout)
    add('FinalNormScale',model.finalnorm.weight); add('FinalNormBias',model.finalnorm.bias)
    linear('Output',model.behavior); linear('Attention',model.attention)
    count=sum(p.numel() for p in model.parameters())
    assert count==TRAINED_PARAMETERS,count
    lines=['// Copyright (c) 2026 Frame Bridge contributors.','// SPDX-License-Identifier: Apache-2.0',
        '// Generated by tools/train_kitty_policy.py from original synthetic temporal episodes.',
        f'// Genuine trained causal Transformer: 2 pre-LN blocks, width{WIDTH}, {HEADS} heads, FF{FF}, context{CONTEXT}.',
        f'// {count} trained float32 parameters + {CONTEXT*WIDTH} fixed positional values. No pretrained data.',
        '// Synthetic teacher agreement is not a measure of real-cat fidelity or consciousness.',
        'namespace BabbleFrameBridge.App;','internal static class KittyModel','{',
        f'    internal const int ParameterCount = {count};']
    for name,array in arrays:
        lines+=['    internal static readonly float[] '+name+' =','    [']
        flat=array.ravel()
        for start in range(0,len(flat),8):
            lines.append('        '+', '.join(format(float(v),'.9g')+'f' for v in flat[start:start+8])+',')
        lines.append('    ];')
    lines+=['}','']
    destination=root/'src/BabbleFrameBridge.App/KittyModel.cs'
    write_atomic(destination,'\n'.join(lines))
    cases=[]
    for name,x in probe_sequences().items():
        a,g=predictions(model,x)
        cases.append({'name':name,'observations':x.tolist(),'behavior':a.tolist(),'attention':g.tolist()})
    rng=np.random.default_rng(SEED+77)
    for length in (1,2,8,32,45):
        x=rng.random((length,INPUTS),dtype=np.float32)
        a,g=predictions(model,x)
        cases.append({'name':'bounded_random_'+str(length),'observations':x.tolist(),'behavior':a.tolist(),'attention':g.tolist()})
    fixture={'architecture':ARCHITECTURE,'float_tolerance':.0001,
        'input_names':INPUT_NAMES,'behavior_names':NAMES,'attention_names':ATTENTION,'cases':cases}
    write_atomic(root/'tests/Kitty.Tests/transformer-golden.json',json.dumps(fixture,indent=2)+'\n')
    report['model_sha256']=hashlib.sha256(destination.read_bytes()).hexdigest()
    write_atomic(root/'docs/RELAY-TRAINING.json',json.dumps(report,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir',type=Path,default=Path('work/relay-transformer-training'))
    parser.add_argument('--episodes',type=int,default=8000)
    parser.add_argument('--validation-episodes',type=int,default=1200)
    parser.add_argument('--epochs',type=int,default=90)
    parser.add_argument('--batch-size',type=int,default=96)
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args()
    torch.set_num_threads(args.threads); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SEED); np.random.seed(SEED)
    args.work_dir.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parent.parent
    train=episodes(SEED,args.episodes)
    valid=episodes(SEED+1000,args.validation_episodes)
    audit=episodes(SEED+5000,args.validation_episodes)
    unseen=episodes(SEED+6000,600,mixes=True,only_scenarios=[22,23,24,25])
    x,a,g,mask,_=train
    tensor=[torch.from_numpy(v) for v in (x,a,g,mask)]
    model=Policy()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0018,weight_decay=.002)
    first_epoch=0
    rng=np.random.default_rng(SEED+3)
    start=time.perf_counter(); history=[]
    best_score=float('inf'); best_epoch=first_epoch
    for epoch in range(first_epoch,args.epochs):
        order=rng.permutation(len(x)); losses=[]
        lr=.0018*(.25+.75*.5*(1+math.cos(math.pi*epoch/args.epochs)))
        for group in optimizer.param_groups: group['lr']=lr
        model.train()
        for begin in range(0,len(x),args.batch_size):
            indices=order[begin:begin+args.batch_size]
            xx,aa,gg,vv=(v[indices] for v in tensor)
            pa,pg=model(xx)
            behavior=-(aa*pa.log_softmax(-1)).sum(-1)
            attention=-(gg*pg.log_softmax(-1)).sum(-1)
            loss=((behavior+.6*attention)*vv).sum()/vv.sum()
            optimizer.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step(); losses.append(float(loss.detach()))
        if (epoch+1)%10==0 or epoch==0 or epoch+1==args.epochs:
            model.eval(); score=metrics(model,valid)
            record={'epoch':epoch+1,'loss':float(np.mean(losses)),
                'behavior_agreement':score['behavior']['top1_agreement'],
                'attention_agreement':score['attention']['top1_agreement'],
                'behavior_kl':score['behavior']['mean_kl_nats'],
                'attention_kl':score['attention']['mean_kl_nats'],
                'elapsed_seconds':time.perf_counter()-start}
            selection_score=record['behavior_kl']+.6*record['attention_kl']
            if selection_score<best_score:
                best_score=selection_score; best_epoch=epoch+1
                torch.save({'model':model.state_dict(),'epoch':best_epoch,'selection_score':best_score},args.work_dir/'best.pt')
            history.append(record); print(json.dumps(record),flush=True)
            torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch+1},args.work_dir/'checkpoint.pt')
    selected=torch.load(args.work_dir/'best.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(selected['model'])
    model.eval(); audit_start=time.perf_counter()
    heldout=metrics(model,audit); mixed=metrics(model,unseen)
    audit_seconds=time.perf_counter()-audit_start
    temporal=sequence_checks(model)
    assert heldout['behavior']['top1_agreement']>.78,heldout['behavior']
    assert heldout['attention']['top1_agreement']>.88,heldout['attention']
    report={'architecture':ARCHITECTURE,'seed':SEED,'config':vars(args)|{'work_dir':None,'resume':None},
        'trained_parameters':sum(p.numel() for p in model.parameters()),'fixed_position_values':CONTEXT*WIDTH,
        'numeric_weight_bytes':(TRAINED_PARAMETERS+CONTEXT*WIDTH)*4,'input_count':22,'behavior_count':35,'attention_count':5,
        'context_steps':CONTEXT,'blocks':2,'model_width':WIDTH,'heads':HEADS,'feed_forward_width':FF,
        'selected_epoch':best_epoch,'checkpoint_selection':'lowest behavior KL + 0.6 * attention KL on progress-validation seed only',
        'layer_norm_epsilon':1e-5,'gelu':'tanh approximation','torch_version':torch.__version__,
        'numpy_version':np.__version__,'device':'CPU','training_seconds':time.perf_counter()-start,
        'final_audit_seconds':audit_seconds,
        'training_episodes':args.episodes,'training_valid_timesteps':int(mask.sum()),
        'train_scenarios':list(SCENARIOS[:22]),'progress_validation_seed':SEED+1000,
        'heldout_seed':SEED+5000,'unseen_mix_seed':SEED+6000,
        'heldout':heldout,'unseen_ordered_mixtures':mixed,'temporal_validation':temporal,'history':history,
        'limitations':['Original synthetic teacher labels only; no real-cat dataset or measured feline cognition.',
            'Agreement measures imitation of this teacher, not intelligence, consciousness, or animal fidelity.',
            'Physical eligibility, welfare-style interaction constraints, routines and rendering remain ordinary application code.',
            'Context is bounded to32 observations; longer needs are supplied by existing explicit need/memory inputs.',
            'No user recordings, cloud service, pretrained model, audio output, or language generation.']}
    export(model,root,report)
    print(json.dumps({'exported':True,'trained_parameters':report['trained_parameters'],'heldout':heldout['behavior'],
        'attention':heldout['attention'],'training_seconds':report['training_seconds']}),flush=True)


if __name__=='__main__': main()
