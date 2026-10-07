"""Dense baseline, L2 and dropout with training-only scaling and held-out evaluation."""
import argparse,json
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
from regularization_models import build_baseline,build_regularized_model_1,build_regularized_model_2
from sequence_experiment import data

def train(output,epochs=30):
    if type(epochs) is not int or not 1<=epochs<=300:raise ValueError('Epochs must be 1..300')
    dataset=load_breast_cancer();x,rest,y,yr=train_test_split(dataset.data,dataset.target,test_size=.4,stratify=dataset.target,random_state=42)
    v,e,yv,ye=train_test_split(rest,yr,test_size=.5,stratify=yr,random_state=43)
    scaler=StandardScaler().fit(x);x,v,e=[scaler.transform(z).astype('float32') for z in [x,v,e]]
    root=Path(output);root.mkdir(parents=True,exist_ok=True);results={}
    for name,build in [('baseline',build_baseline),('l2',build_regularized_model_1),('dropout',build_regularized_model_2)]:
        tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(42);model=build();path=root/(name+'.keras')
        history=model.fit(data(x,y,32).shuffle(16,seed=42),validation_data=data(v,yv,32),epochs=epochs,verbose=2,
            callbacks=[tf.keras.callbacks.ModelCheckpoint(str(path),monitor='val_loss',save_best_only=True),tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=8)])
        model=tf.keras.models.load_model(path);loaded=tf.keras.models.load_model(path)
        prob=model.predict(data(e,ye,32),verbose=0).ravel();np.testing.assert_array_equal(prob,loaded.predict(data(e,ye,32),verbose=0).ravel())
        prediction=(prob>=.5).astype(int);assert np.isfinite(prob).all()
        train_prediction=(model.predict(data(x,y,32),verbose=0).ravel()>=.5).astype(int)
        val_prediction=(model.predict(data(v,yv,32),verbose=0).ravel()>=.5).astype(int)
        train_acc=float(accuracy_score(y,train_prediction));val_acc=float(accuracy_score(yv,val_prediction))
        results[name]={'selected_epoch':int(np.argmin(history.history['val_loss'])+1),'history':{k:[float(z) for z in values] for k,values in history.history.items()},
          'checkpoint_train_accuracy':train_acc,'checkpoint_validation_accuracy':val_acc,'checkpoint_accuracy_gap':train_acc-val_acc,
          'test':{'accuracy':float(accuracy_score(ye,prediction)),'macro_f1':float(f1_score(ye,prediction,average='macro')),'confusion_matrix':confusion_matrix(ye,prediction,labels=[0,1]).tolist()},'checkpoint_reload_exact':True}
    np.savez(root/'scaler.npz',mean=scaler.mean_,scale=scaler.scale_)
    with np.load(root/'scaler.npz',allow_pickle=False) as saved:np.testing.assert_allclose((dataset.data-saved['mean'])/saved['scale'],scaler.transform(dataset.data),atol=1e-12)
    result={'train':len(x),'validation':len(v),'test':len(e),'models':results,'scaler_reload_matches':True,
      'selection':'validation objective (cross-entropy plus L2 penalty where present); test data selects no checkpoint or setting',
      'limitation':'30-epoch maximum and one seeded row split; no 300-epoch historical reproduction, clinical validity or claim that regularization universally improves performance'}
    (root/'regularization-metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8');return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True);p.add_argument('--epochs',type=int,default=30);args=p.parse_args()
    try:train(args.output_dir,args.epochs)
    except (ValueError,OSError) as error:p.exit(2,'Failed: '+str(error)+'\n')
