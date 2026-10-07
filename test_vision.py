import numpy as np
import pytest
import torch
from vision_models import PlainBlock,ResBlock,PlainNet,MiniResNet

def test_residual_identity_path_survives_zero_branch_while_plain_branch_is_zero():
    x=torch.linspace(-1,1,64).reshape(2,2,4,4)
    plain=PlainBlock(2).eval();residual=ResBlock(2).eval()
    with torch.no_grad():
        for model in [plain,residual]:
            for parameter in model.parameters():parameter.zero_()
        assert torch.equal(plain(x),torch.zeros_like(x))
        assert torch.equal(residual(x),x.relu())

@pytest.mark.parametrize('constructor',[PlainNet,MiniResNet])
def test_convolution_heads_accept_cifar_and_eval_does_not_change_running_statistics(constructor):
    torch.manual_seed(8);model=constructor(num_blocks=2,channels=8,num_classes=10);probe=torch.randn(3,3,32,32)
    logits=model(probe);assert logits.shape==(3,10) and torch.isfinite(logits).all()
    logits.sum().backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    model.eval();before={name:value.clone() for name,value in model.state_dict().items() if 'running_' in name}
    with torch.no_grad():model(probe*100)
    for name,value in before.items():assert torch.equal(value,model.state_dict()[name])

def test_image_inference_rejects_missing_and_malformed_inputs_before_model_load(tmp_path):
    from vision_experiment import predict
    path=tmp_path/'bad.png';path.write_bytes(b'not an image')
    with pytest.raises(ValueError,match='Invalid image'):predict(tmp_path,'plain',path)
    with pytest.raises(ValueError):predict(tmp_path,'absent',path)
