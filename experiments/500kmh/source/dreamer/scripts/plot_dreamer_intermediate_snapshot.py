"""Descriptive snapshot of completed episodes, including unfinished pilots."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from report_dreamer_fixed_center_matched import records,grouped
from matched_curve_reference import PPO, validate_ppo, reference_for, draw_reference

p=argparse.ArgumentParser()
p.add_argument('--run-dir',type=Path,required=True)
p.add_argument('--output-dir',type=Path,required=True)
p.add_argument('--ppo-dir',type=Path,default=None,
               help='Optional PPO run for a comparison curve')
a=p.parse_args()
episodes=records(a.run_dir/'episode_terminal_rates.csv')
meta=json.loads((a.run_dir/'experiment.json').read_text())
amount=int(meta.get('num_envs',20))
steps={int(r['steps']) for r in episodes}
valid={s for s in steps if sum(int(r['steps'])==s for r in episodes)==amount}
episodes=[r for r in episodes if int(r['steps']) in valid]
maximum=max(valid)
if a.ppo_dir is not None:
    validate_ppo(meta,a.ppo_dir)
# A task may preserve the reachable box while changing decision granularity.
# Draw an oracle line only when reference metadata matches this task exactly.
try:
    reference=reference_for(meta)
except AssertionError:
    reference=None
x,y,low,high=grouped(episodes,maximum)
assert np.all(np.diff(x)>0)
a.output_dir.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':14,'pdf.fonttype':42})
fig,ax=plt.subplots(figsize=(8.4,5.4))
ax.plot(x,y,color='#2462ce',linewidth=2,label='Proposed Alg. 1')
if a.ppo_dir is not None:
    px,py,_,_=grouped(records(a.ppo_dir/'episode_terminal_rates.csv'),maximum)
    ax.plot(px,py,color='#d93635',linewidth=2,label='PPO')
draw_reference(ax,reference)
if amount == 1:
    ax.legend(loc='best',framealpha=.95)
else:
    ax.legend(loc='lower center',bbox_to_anchor=(.58,.05),framealpha=.9)
ax.set(xlim=(0,maximum),xlabel='Steps',ylabel='AFDM rate (bps/Hz)',title='Training curves')
ax.xaxis.set_major_formatter(FuncFormatter(lambda v,_:'0' if v==0 else f'{v/1000:g}K'))
ax.grid(alpha=.2)
ax.tick_params(direction='in',top=True,right=True)
fig.tight_layout()
fig.savefig(a.output_dir/'training_snapshot.pdf',bbox_inches='tight')
fig.savefig(a.output_dir/'training_snapshot.png',dpi=180,bbox_inches='tight')
(a.output_dir/'snapshot_metadata.json').write_text(json.dumps(dict(source_run=str(a.run_dir),
    completed_episode_count=len(episodes),through_step=maximum,
    num_envs=amount,curve_points=len(x),
    terminal_record_interval=int(meta['environment']['max_steps'])*amount,
    metric='actual episode terminal rate' if amount==1 else '20-episode terminal mean',
    ppo_metric='optional comparison; absent in portable standalone runs',
    run_complete=(a.run_dir/'training_complete.json').exists(),
    shade='none',
    exhaustive_reference=reference,
    ppo_source_run=str(a.ppo_dir) if a.ppo_dir else None,
    final_batch_mean=float(y[-1])),indent=2))
print(maximum,float(y[-1]))
