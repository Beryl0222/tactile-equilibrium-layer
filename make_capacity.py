"""按固定种子生成容量输入，不计算算子输出。"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
BASE=Path(__file__).resolve().parent

def generate(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(261126)
    arrays={};meta={'format_version':1,'source':'本地合成容量输入','seed':261126}
    n=2048;deg=8;d=16
    arrays['X']=rng.normal(0,.1,(n,4))
    arrays['edge_u']=np.repeat(np.arange(n),deg)
    arrays['edge_v']=(arrays['edge_u']+np.tile(np.arange(1,deg+1),n))%n
    arrays['edge_weight']=np.full(n*deg,1/(deg*2))
    arrays['W']=np.eye(d)*.2;arrays['U']=rng.normal(0,.1,(4,d));arrays['b']=np.zeros(d)
    meta.update(node_ids=['node-'+str(i) for i in range(n)],revision=1,max_actions=200)
    np.savez(output/'arrays.npz',**arrays)
    meta['arrays']={k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in arrays.items()}
    (output/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return meta

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='生成可复现的合成容量输入')
    parser.add_argument('--output',required=True)
    args=parser.parse_args();meta=generate(args.output)
    print(json.dumps({'数组数':len(meta['arrays']),'资料SHA256':hashlib.sha256((Path(args.output)/'arrays.npz').read_bytes()).hexdigest()},ensure_ascii=False))
