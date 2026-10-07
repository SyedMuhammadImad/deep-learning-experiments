"""Plain/residual CIFAR subset training with disjoint official test and saved local inference."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset,DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
from vision_models import PlainNet,MiniResNet

MODELS={'plain':PlainNet,'residual':MiniResNet}
def balanced(labels,count,seed):
    rng=np.random.default_rng(seed);ids=[];labels=np.asarray(labels)
    for label in np.unique(labels):ids.extend(rng.choice(np.flatnonzero(labels==label),count//10,replace=False))
    rng.shuffle(ids);return np.asarray(ids)

def evaluate(model,x,y):
    model.eval();probabilities=[]
    with torch.no_grad():
        for (batch,) in DataLoader(TensorDataset(x),batch_size=64):probabilities.append(model(batch).softmax(dim=1))
    probability=torch.cat(probabilities).numpy();prediction=probability.argmax(axis=1)
    if not np.isfinite(probability).all():raise ValueError('Non-finite output')
    loss=float(-np.log(np.maximum(probability[np.arange(len(y)),y],1e-30)).mean())
    return loss,probability,{'accuracy':float(accuracy_score(y,prediction)),'macro_f1':float(f1_score(y,prediction,average='macro')),'confusion_matrix':confusion_matrix(y,prediction,labels=list(range(10))).tolist()}

def train(data_dir,output,epochs=5):
    if type(epochs) is not int or not 1<=epochs<=50:raise ValueError('Epochs must be 1..50')
    from torchvision.datasets import CIFAR10
    a=CIFAR10(data_dir,train=True,download=False);b=CIFAR10(data_dir,train=False,download=False)
    ai=balanced(a.targets,5000,42);bi=balanced(b.targets,1000,43);raw=a.data[ai];labels=np.asarray(a.targets)[ai];test=b.data[bi];test_y=np.asarray(b.targets)[bi]
    raw,val,y,yv=train_test_split(raw,labels,test_size=1000,stratify=labels,random_state=44)
    convert=lambda x:torch.tensor(x.transpose(0,3,1,2).copy(),dtype=torch.float32)/255
    x,v,e=map(convert,[raw,val,test]);mean=x.mean(dim=(0,2,3));std=x.std(dim=(0,2,3),unbiased=False).clamp_min(1e-6)
    x,v,e=[(value-mean[None,:,None,None])/std[None,:,None,None] for value in [x,v,e]]
    output=Path(output);output.mkdir(parents=True,exist_ok=True);results={}
    for name,constructor in MODELS.items():
        torch.manual_seed(42);model=constructor(num_blocks=3,channels=32,num_classes=10);optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        criterion=nn.CrossEntropyLoss();history=[];best=np.inf;generator=torch.Generator().manual_seed(42)
        loader=DataLoader(TensorDataset(x,torch.tensor(y,dtype=torch.long)),batch_size=64,shuffle=True,generator=generator)
        for epoch in range(epochs):
            model.train();total=0.;seen=0
            for batch,target in loader:
                # Augmentation exists only here; validation/test tensors cannot share transform mutation.
                flip=torch.rand(len(batch))<.5;batch=batch.clone();batch[flip]=batch[flip].flip(-1)
                optimizer.zero_grad();loss=criterion(model(batch),target)
                if not torch.isfinite(loss):raise ValueError('Training diverged')
                loss.backward();optimizer.step();total+=loss.item()*len(batch);seen+=len(batch)
            val_loss,_,val_metrics=evaluate(model,v,yv);history.append({'epoch':epoch+1,'train_loss':total/seen,'validation_loss':val_loss,'validation_accuracy':val_metrics['accuracy']})
            if val_loss<best:
                best=val_loss;selected=epoch+1;torch.save(model.state_dict(),output/(name+'.pt'))
            print(name+' epoch '+str(epoch+1)+' validation '+str(val_metrics['accuracy']),flush=True)
        model.load_state_dict(torch.load(output/(name+'.pt'),weights_only=True,map_location='cpu'));_,probability,metrics=evaluate(model,e,test_y)
        loaded=constructor(num_blocks=3,channels=32,num_classes=10);loaded.load_state_dict(torch.load(output/(name+'.pt'),weights_only=True,map_location='cpu'))
        _,reloaded,_=evaluate(loaded,e,test_y);np.testing.assert_array_equal(probability,reloaded)
        results[name]={'history':history,'selected_epoch':selected,'test':metrics,'checkpoint_reload_exact':True,'parameters':sum(p.numel() for p in model.parameters())}
    metadata={'mean':mean.tolist(),'std':std.tolist(),'classes':a.classes,'size':32,'num_blocks':3,'channels':32}
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    result={'train':len(x),'validation':len(v),'official_test':len(e),'image_size':32,'models':results,
        'data_fingerprint':hashlib.sha256(raw.tobytes()+y.tobytes()+val.tobytes()+yv.tobytes()+test.tobytes()+test_y.tobytes()).hexdigest(),
        'selection':'validation cross-entropy; normalization fits training pixels; augmentation only on training batches',
        'limitation':'five-epoch balanced CIFAR subsets; no full benchmark, Cats/Dogs or six-landscape dataset reproduction claim'}
    (output/'vision-metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8');return result

def predict(model_dir,architecture,image):
    if architecture not in MODELS:raise ValueError('Unknown architecture')
    from PIL import Image,UnidentifiedImageError
    root=Path(model_dir);path=Path(image)
    if path.stat().st_size>5_000_000:raise ValueError('Image file must be under 5 MB')
    try:
        with Image.open(path) as source:
            if source.width*source.height>16_000_000:raise ValueError('Image dimensions too large')
            pixels=np.asarray(source.convert('RGB').resize((32,32)),dtype=np.float32).copy()/255
    except UnidentifiedImageError as error:raise ValueError('Invalid image') from error
    metadata=json.loads((root/'metadata.json').read_text());model=MODELS[architecture](num_blocks=metadata['num_blocks'],channels=metadata['channels'],num_classes=len(metadata['classes']))
    model.load_state_dict(torch.load(root/(architecture+'.pt'),weights_only=True,map_location='cpu'));model.eval()
    tensor=torch.tensor(pixels.transpose(2,0,1))[None,:]
    tensor=(tensor-torch.tensor(metadata['mean'])[None,:,None,None])/torch.tensor(metadata['std'])[None,:,None,None]
    with torch.no_grad():probability=model(tensor).softmax(dim=1)[0].numpy()
    if not np.isfinite(probability).all():raise ValueError('Invalid checkpoint normalization or probabilities')
    return {'label':metadata['classes'][int(probability.argmax())],'uncalibrated_probabilities':probability.tolist()}

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    t=sub.add_parser('train');t.add_argument('--data-dir',required=True);t.add_argument('--output-dir',required=True);t.add_argument('--epochs',type=int,default=5)
    i=sub.add_parser('predict');i.add_argument('--model-dir',required=True);i.add_argument('--architecture',choices=list(MODELS),required=True);i.add_argument('--image',required=True)
    args=p.parse_args()
    try:
        if args.command=='train':train(args.data_dir,args.output_dir,args.epochs)
        else:print(json.dumps(predict(args.model_dir,args.architecture,args.image)))
    except (ValueError,OSError,KeyError) as error:p.exit(2,'Failed: '+str(error)+'\n')
if __name__=='__main__':main()
