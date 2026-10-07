import numpy as np
import pytest
tf=pytest.importorskip('tensorflow')
from sequence_experiment import build_model,token_array

@pytest.mark.parametrize('architecture',['rnn','lstm','gru'])
def test_padding_does_not_change_prediction_and_saved_model_reloads(architecture,tmp_path):
    tf.keras.utils.set_random_seed(4);model=build_model(architecture,30,8)
    short=np.array([[1,4,5,2]]);padded=np.array([[1,4,5,2,0,0,0]])
    expected=model(short,training=False).numpy();np.testing.assert_allclose(expected,model(padded,training=False).numpy(),atol=1e-7)
    model.train_on_batch(np.array([[1,4,5,2],[1,3,8,2]]),np.array([0,1]));path=tmp_path/(architecture+'.keras');model.save(path)
    loaded=tf.keras.models.load_model(path);np.testing.assert_allclose(model(short,training=False).numpy(),loaded(short,training=False).numpy(),atol=1e-7)

def test_token_inference_rejects_invalid_empty_and_noninteger_ids():
    for tokens in [[],[0,0],[-1,2],[1.0,2],[True,2],[10000],list(range(501))]:
        with pytest.raises(ValueError):token_array(tokens)
    assert token_array([1,2,3]).shape==(1,100)
    assert token_array([1,2,3])[0,:4].tolist()==[1,2,3,0]
