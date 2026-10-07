"""Fresh normalization-aware tabular training; checkpoint selection uses validation only."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
from numpy_layers import *

def save_model(path,model):
    data={'config':np.array(json.dumps({'use_bn':model.use_bn,'lr':model.lr,'input_dim':model.dims[0],'hidden_dims':model.dims[1:-1]}))}
    for i,(w,b) in enumerate(zip(model.W,model.b)):data.update({f'W{i}':w,f'b{i}':b})
    for i,layer in enumerate(model.bn_layers):
        for name in ['gamma','beta','running_mean','running_var']:data[f'{name}{i}']=getattr(layer,name)
    np.savez(path,**data)

def load_model(path):
    if Path(path).stat().st_size>100_000_000:raise ValueError('Model file too large')
    with np.load(path,allow_pickle=False) as data:
        config=json.loads(str(data['config']))
        if config['input_dim']>10000 or len(config['hidden_dims'])>10 or any(d>1000 for d in config['hidden_dims']):raise ValueError('Model dimensions too large')
        model=MLPWithBN(**config)
        for i in range(len(model.W)):
            for name in ['W','b']:
                value=data[f'{name}{i}'].copy();expected=getattr(model,name)[i]
                if value.shape!=expected.shape or not np.isfinite(value).all():raise ValueError('Invalid checkpoint parameter')
                getattr(model,name)[i]=value
        for i,layer in enumerate(model.bn_layers):
            for name in ['gamma','beta','running_mean','running_var']:
                value=data[f'{name}{i}'].copy()
                if value.shape!=(layer.n_features,) or not np.isfinite(value).all() or (name=='running_var' and (value<0).any()):raise ValueError('Invalid normalization checkpoint')
                setattr(layer,name,value)
    return model

def compare(frame,y,output,epochs=150):
    xt,rest,yt,yr=train_test_split(frame,np.asarray(y),test_size=.4,stratify=y,random_state=42)
    xv,xe,yv,ye=train_test_split(rest,yr,test_size=.5,stratify=yr,random_state=43)
    numeric=xt.select_dtypes(include=['number']).columns.tolist();categorical=[c for c in xt if c not in numeric]
    prepare=ColumnTransformer([('numeric',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),numeric),
        ('category',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),categorical)])
    x=prepare.fit_transform(xt);v=prepare.transform(xv);e=prepare.transform(xe)
    output=Path(output);output.mkdir(parents=True,exist_ok=True);results={}
    for use_bn in [False,True]:
        name='batch_normalized' if use_bn else 'plain';model=MLPWithBN(use_bn=use_bn,lr=.02,input_dim=x.shape[1]);rng=np.random.default_rng(42)
        history=[];best=np.inf;unimproved=0;path=output/(name+'.npz')
        for epoch in range(epochs):
            order=rng.permutation(len(x))
            for start in range(0,len(x),64):
                indices=order[start:start+64];model.step(x[indices],yt[indices])
            train_loss=model.compute_loss(x,yt);val_loss=model.compute_loss(v,yv)
            if not np.isfinite([train_loss,val_loss]).all():raise ValueError('Non-finite loss')
            history.append({'epoch':epoch+1,'train_loss':train_loss,'validation_loss':val_loss})
            if val_loss<best:best=val_loss;save_model(path,model);selected=epoch+1;unimproved=0
            else:unimproved+=1
            if unimproved>=20:break
        model=load_model(path);probability=model.forward(e,False).ravel();prediction=(probability>=.5).astype(int)
        loaded=load_model(path);np.testing.assert_array_equal(probability,loaded.forward(e,False).ravel())
        results[name]={'selected_epoch':selected,'history':history,'test':{'accuracy':float(accuracy_score(ye,prediction)),
            'macro_f1':float(f1_score(ye,prediction,average='macro')),'confusion_matrix':confusion_matrix(ye,prediction,labels=[0,1]).tolist()},'checkpoint_reload_exact':True}
    # Only trusted local preprocessing files may be reloaded; none are committed.
    import joblib
    joblib.dump(prepare,output/'preprocessor.joblib');np.testing.assert_array_equal(e,joblib.load(output/'preprocessor.joblib').transform(xe))
    return {'train':len(x),'validation':len(v),'test':len(e),'features_after_encoding':x.shape[1],'models':results,'training_only_preprocessing_reload_exact':True}

def main():
    p=argparse.ArgumentParser();p.add_argument('--heart-csv',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--epochs',type=int,default=150)
    args=p.parse_args()
    try:
        if not 1<=args.epochs<=500:raise ValueError('Epochs must be 1..500')
        root=Path(args.output_dir);root.mkdir(parents=True,exist_ok=True);breast=load_breast_cancer(as_frame=True)
        heart=pd.read_csv(args.heart_csv)
        if 'num' not in heart or heart['num'].isna().any():raise ValueError('Expected finite original heart num labels')
        y=(pd.to_numeric(heart.pop('num'),errors='raise')>0).astype(int).to_numpy();heart=heart.drop(columns=['id','dataset'],errors='ignore')
        result={'breast_cell':compare(breast.data,breast.target,root/'breast',args.epochs),'heart_table':compare(heart,y,root/'heart',args.epochs),
          'selection':'same seeded initialization/order; validation cross-entropy selects checkpoint; no test-driven epoch selection',
          'limitation':'one random row split; combined heart provenance/cohort separation unverified; offline experiment, not clinical validation'}
        (root/'numeric-metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
        print('Fresh plain and correctly backpropagated normalization models completed on both tables.',flush=True)
    except (ValueError,OSError,KeyError) as error:p.exit(2,'Failed: '+str(error)+'\n')
if __name__=='__main__':main()
