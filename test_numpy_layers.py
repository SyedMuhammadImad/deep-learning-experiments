import numpy as np
import pytest
import torch
from numpy_layers import *

@pytest.mark.parametrize('kind',['batch','layer'])
def test_normalization_forward_input_and_affine_gradients_match_autograd(kind):
    rng=np.random.default_rng(5);x=rng.normal(size=(7,4));upstream=rng.normal(size=x.shape)
    layer=BatchNorm1d(4) if kind=='batch' else LayerNorm1d(4)
    layer.gamma=rng.normal(size=4);layer.beta=rng.normal(size=4)
    output=layer.forward(x);dx,dg,db=layer.backward(upstream)
    tx=torch.tensor(x,requires_grad=True);gamma=torch.tensor(layer.gamma,requires_grad=True);beta=torch.tensor(layer.beta,requires_grad=True)
    axis=0 if kind=='batch' else 1
    normalized=(tx-tx.mean(dim=axis,keepdim=True))/torch.sqrt(tx.var(dim=axis,unbiased=False,keepdim=True)+layer.eps)
    reference=normalized*gamma+beta;(reference*torch.tensor(upstream)).sum().backward()
    np.testing.assert_allclose(output,reference.detach().numpy(),atol=1e-12)
    for actual,expected in [(dx,tx.grad.numpy()),(dg,gamma.grad.numpy()),(db,beta.grad.numpy())]:np.testing.assert_allclose(actual,expected,atol=1e-11)

def test_batch_inference_uses_frozen_statistics_and_constant_inputs_are_finite():
    layer=BatchNorm1d(3,momentum=.2);layer.forward(np.array([[1.,2.,3.],[3.,4.,5.]]));mean=layer.running_mean.copy();variance=layer.running_var.copy()
    probe=np.ones((2,3))*100;output=layer.forward(probe,False)
    np.testing.assert_allclose(output,(probe-mean)/np.sqrt(variance+layer.eps))
    np.testing.assert_array_equal(layer.running_mean,mean)
    with pytest.raises(ValueError):layer.backward(probe)
    assert np.isfinite(layer.forward(np.ones((2,3)))).all()
    assert np.isfinite(LayerNorm1d(3).forward(np.ones((2,3)))).all()

@pytest.mark.parametrize('use_bn',[False,True])
def test_entire_network_gradients_match_finite_differences_for_all_parameters(use_bn):
    rng=np.random.default_rng(6);x=rng.normal(size=(8,2));y=np.arange(8)%2
    model=MLPWithBN(use_bn=use_bn,input_dim=2,hidden_dims=(3,2));model.forward(x,True);gradient=model.gradients(y)
    groups={'W':model.W,'b':model.b,'gamma':[layer.gamma for layer in model.bn_layers],'beta':[layer.beta for layer in model.bn_layers]}
    for name,parameters in groups.items():
        for j,parameter in enumerate(parameters):
            numerical=np.zeros_like(parameter)
            for index in np.ndindex(parameter.shape):
                original=parameter[index];parameter[index]=original+1e-6;plus=model.compute_loss(x,y,True)
                parameter[index]=original-1e-6;minus=model.compute_loss(x,y,True);parameter[index]=original
                numerical[index]=(plus-minus)/2e-6
            np.testing.assert_allclose(gradient[name][j],numerical,atol=1e-6,rtol=1e-5)

def test_normalization_affine_parameters_update_and_unfit_gradient_rejected():
    rng=np.random.default_rng(4);x=rng.normal(size=(12,2));y=(x[:,0]>0).astype(int)
    model=MLPWithBN(input_dim=2,hidden_dims=(4,));before=model.bn_layers[0].gamma.copy();model.step(x,y)
    assert not np.array_equal(before,model.bn_layers[0].gamma)
    model.forward(x,False)
    with pytest.raises(ValueError):model.gradients(y)
    with pytest.raises(ValueError):model.step(x,np.full(12,2))

def test_scalar_adam_matches_independent_library_and_loss_gradient_reference():
    actual=AdamOptimizer(lr=.03);w=1.2;reference=torch.tensor(1.2,dtype=torch.float64,requires_grad=True);optimizer=torch.optim.Adam([reference],lr=.03)
    for grad in [3.,-2.,.4,0.,1.]:
        w=actual.step(w,grad);reference.grad=torch.tensor(grad,dtype=torch.float64);optimizer.step()
        assert abs(w-reference.item())<1e-12
    for w in [-2.,0.,1.2,5.]:assert abs(grad_fn(w)-(loss_fn(w+1e-6)-loss_fn(w-1e-6))/2e-6)<1e-8

def test_invalid_normalization_and_network_settings_fail_explicitly():
    for construct in [lambda:BatchNorm1d(0),lambda:BatchNorm1d(2,eps=0),lambda:LayerNorm1d(2,eps=-1),lambda:MLPWithBN(lr=0),lambda:AdamOptimizer(beta1=1)]:
        with pytest.raises(ValueError):construct()
    with pytest.raises(ValueError):BatchNorm1d(2).forward(np.ones((2,3)))
    with pytest.raises(ValueError):LayerNorm1d(2).forward(np.full((2,2),np.nan))

def test_invalid_target_does_not_mutate_model_statistics_or_parameters():
    model=MLPWithBN(input_dim=2,hidden_dims=(3,));before=[p.copy() for p in model.W+model.b]
    mean=model.bn_layers[0].running_mean.copy();variance=model.bn_layers[0].running_var.copy()
    with pytest.raises(ValueError):model.step(np.ones((4,2)),np.full(4,2))
    for actual,expected in zip(model.W+model.b,before):np.testing.assert_array_equal(actual,expected)
    np.testing.assert_array_equal(model.bn_layers[0].running_mean,mean);np.testing.assert_array_equal(model.bn_layers[0].running_var,variance)
    with pytest.raises(ValueError):model.accuracy(np.ones((4,2)),[1])

def test_checkpoint_reloads_all_normalization_statistics_and_rejects_bad_shape(tmp_path):
    from numeric_experiment import save_model,load_model
    x=np.array([[1.,2.],[2.,-1.],[-1.,3.],[4.,2.]]);y=np.array([0,1,0,1])
    model=MLPWithBN(input_dim=2,hidden_dims=(3,));model.step(x,y);path=tmp_path/'weights.npz';save_model(path,model)
    np.testing.assert_array_equal(model.forward(x,False),load_model(path).forward(x,False))
    with np.load(path,allow_pickle=False) as archive:data={key:archive[key] for key in archive.files}
    data['W0']=np.zeros((1,1));np.savez(path,**data)
    with pytest.raises(ValueError,match='parameter'):load_model(path)

def test_momentum_matches_scaled_library_momentum_and_rmsprop_exact_ema():
    model=MomentumOptimizer(lr=.03,beta=.9);w=1.2
    parameter=torch.tensor(1.2,dtype=torch.float64,requires_grad=True);reference=torch.optim.SGD([parameter],lr=.03*(1-.9),momentum=.9)
    rms=RMSPropOptimizer(lr=.03,rho=.9,eps=1e-5);value=1.2;gradients=[3.,-2.,.4,0.,1.]
    for i,g in enumerate(gradients):
        w=model.step(w,g);parameter.grad=torch.tensor(g,dtype=torch.float64);reference.step();assert abs(w-parameter.item())<1e-12
        expected_variance=(1-.9)*sum(.9**(i-j)*v*v for j,v in enumerate(gradients[:i+1]))
        expected=value-.03*g/np.sqrt(expected_variance+1e-5);value=rms.step(value,g);assert abs(value-expected)<1e-12
    with pytest.raises(ValueError):MomentumOptimizer(beta=1)
    with pytest.raises(ValueError):RMSPropOptimizer(eps=0)
