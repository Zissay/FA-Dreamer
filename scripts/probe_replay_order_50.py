"""CPU-only sampling diagnostic; original prefill data never reaches fresh runs."""
from pathlib import Path
import argparse,sys,json
sys.modules['tensorflow']=None
root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--prefix-file',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
sys.path.insert(0,str(root/'vendor/runtime'))
from dreamerv3 import embodied as e
import numpy as np
stored=np.load(args.prefix_file)
prefix={k:stored[k][:,:115] for k in stored.files}
assert int(prefix['is_terminal'].sum())==40
assert int(prefix['is_first'].sum())==60
results=[]
for replica in range(3):
    replay=e.replay.Uniform(64,200000,directory=None,seed=0)
    for t in range(115):
        for worker in range(20):replay.add({k:v[worker,t] for k,v in prefix.items()},worker)
    samples={}
    # Identify every eligible sequence using its raw id array; identifiers are
    # fixed input metadata here and are not used as model observations.
    for seq in replay.table.values():
        ids=np.stack([step['id'] for step in seq]);samples[ids.tobytes()]=len(samples)
    dataset=e.Prefetch([replay.dataset for _ in range(16)],workers=2)
    batches=[]
    for _ in range(32):
        batch=next(iter(dataset)) if not batches else next(dataset)
        batches.append([samples[ids.tobytes()] for ids in batch['id']])
    dataset._running=False
    results.append(batches)
report={'diagnostic_only':True,'training_updates':0,'new_environment_actions':0,
    'source':'Identical historical first115 callback frames per20 workers; contains only uniform-prefill observations',
    'sampler_seed_all':0,'data_loaders_all':2,'batch_size_all':16,'batches_per_replica':32,
    'sequence_indices_by_batch':results,
    'unequal_batch_slots_replica1_vs2':int(np.sum(np.array(results[0])!=np.array(results[1]))),
    'unequal_batch_slots_replica1_vs3':int(np.sum(np.array(results[0])!=np.array(results[2]))),
    'scope':'Direct sampling-order variability probe; does not establish cause of final-performance gap'}
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='sequence_indices_by_batch'},indent=2))
