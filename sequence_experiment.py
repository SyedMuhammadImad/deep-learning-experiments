"""SimpleRNN/LSTM/GRU comparison on official pre-indexed IMDB benchmark subsets."""
import argparse,json
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix

def build_model(architecture,vocabulary=10000,units=32):
    if architecture not in ['rnn','lstm','gru'] or type(vocabulary) is not int or vocabulary<4 or type(units) is not int or units<1:raise ValueError('Invalid recurrent model settings')
    layer={'rnn':tf.keras.layers.SimpleRNN,'lstm':tf.keras.layers.LSTM,'gru':tf.keras.layers.GRU}[architecture]
    model=tf.keras.Sequential([tf.keras.Input(shape=(None,),dtype='int64'),tf.keras.layers.Embedding(vocabulary,32,mask_zero=True),layer(units),tf.keras.layers.Dense(1,activation='sigmoid')])
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy']);return model

def data(x,y,batch=64):
    options=tf.data.Options();options.threading.private_threadpool_size=2;options.threading.max_intra_op_parallelism=2
    return tf.data.Dataset.from_tensor_slices((x,y)).with_options(options).batch(batch).prefetch(1)

def token_array(tokens,vocabulary=10000,length=100):
    if not isinstance(tokens,list) or not tokens or len(tokens)>500 or any(type(t) is not int or not 0<=t<vocabulary for t in tokens):raise ValueError('Expected 1..500 integer token IDs within vocabulary')
    if not any(t>0 for t in tokens):raise ValueError('All-padding input is not a review')
    return tf.keras.utils.pad_sequences([tokens],maxlen=length,padding='post',truncating='post')

def train(output,epochs=4):
    if type(epochs) is not int or not 1<=epochs<=30:raise ValueError('Epochs must be 1..30')
    # Keras verifies the official publisher's checksum before loading its legacy array archive.
    (raw,y),(test_raw,test_y)=tf.keras.datasets.imdb.load_data(num_words=10000)
    x,_,y,_=train_test_split(raw,y,train_size=4800,stratify=y,random_state=42)
    tx,_,ty,_=train_test_split(test_raw,test_y,train_size=1000,stratify=test_y,random_state=43)
    x=tf.keras.utils.pad_sequences(x,maxlen=100,padding='post',truncating='post')
    tx=tf.keras.utils.pad_sequences(tx,maxlen=100,padding='post',truncating='post')
    x,v,y,yv=train_test_split(x,y,test_size=800,stratify=y,random_state=44)
    trainset,valset,testset=data(x,y),data(v,yv),data(tx,ty)
    output=Path(output);output.mkdir(parents=True,exist_ok=True);results={}
    for architecture in ['rnn','lstm','gru']:
        tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(42);model=build_model(architecture);path=output/(architecture+'.keras')
        history=model.fit(trainset.shuffle(max(1,len(x)//64),seed=42),validation_data=valset,epochs=epochs,verbose=2,
            callbacks=[tf.keras.callbacks.ModelCheckpoint(str(path),monitor='val_loss',save_best_only=True),tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=2)])
        model=tf.keras.models.load_model(path);prob=model.predict(testset,verbose=0).ravel();prediction=(prob>=.5).astype(int)
        loaded=tf.keras.models.load_model(path);np.testing.assert_allclose(prob,loaded.predict(testset,verbose=0).ravel(),atol=1e-7)
        results[architecture]={'selected_epoch':int(np.argmin(history.history['val_loss'])+1),'history':history.history,'checkpoint_reload_matches':True,
            'test':{'accuracy':float(accuracy_score(ty,prediction)),'macro_f1':float(f1_score(ty,prediction,average='macro')),'confusion_matrix':confusion_matrix(ty,prediction,labels=[0,1]).tolist()}}
        print('Completed '+architecture+' saved-model comparison.',flush=True)
    result={'train':len(x),'validation':len(v),'official_test':len(tx),'maximum_length':100,'vocabulary_limit':10000,'models':results,
        'encoding':'official Keras IMDB word indices, not a newly fitted vocabulary; review prefixes truncated at 100 tokens',
        'selection':'validation loss selects checkpoint; official test subset never selects an epoch',
        'limitation':'one-seed balanced subsets; provided token encoding and truncation; no broad sentiment reliability claim'}
    (output/'sequence-metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8');return result

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    t=sub.add_parser('train');t.add_argument('--output-dir',required=True);t.add_argument('--epochs',type=int,default=4)
    i=sub.add_parser('predict');i.add_argument('--model',required=True);i.add_argument('--tokens-json',required=True)
    args=p.parse_args()
    try:
        if args.command=='train':train(args.output_dir,args.epochs)
        else:
            path=Path(args.tokens_json)
            if path.stat().st_size>10000:raise ValueError('Token input too large')
            tokens=token_array(json.loads(path.read_text(encoding='utf-8')));model=tf.keras.models.load_model(args.model)
            score=float(model(tokens,training=False).numpy()[0,0])
            if not np.isfinite(score):raise ValueError('Non-finite prediction')
            print(json.dumps({'positive':bool(score>=.5),'uncalibrated_positive_probability':score}))
    except (ValueError,OSError,KeyError) as error:p.exit(2,'Failed: '+str(error)+'\n')
if __name__=='__main__':main()
