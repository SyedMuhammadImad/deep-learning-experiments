import pytest,numpy as np
tf=pytest.importorskip('tensorflow')
from regularization_models import build_baseline,build_regularized_model_1,build_regularized_model_2

def test_l2_penalty_gradient_and_unregularized_baseline():
    tf.keras.utils.set_random_seed(42);model=build_regularized_model_1(lam=.001)
    kernels=[layer.kernel for layer in model.layers if isinstance(layer,tf.keras.layers.Dense) and layer.kernel_regularizer is not None]
    with tf.GradientTape() as tape:penalty=tf.add_n(model.losses)
    gradients=tape.gradient(penalty,kernels)
    np.testing.assert_allclose(penalty.numpy(),.001*sum(float(tf.reduce_sum(k*k)) for k in kernels),rtol=1e-6)
    for kernel,gradient in zip(kernels,gradients):np.testing.assert_allclose(gradient,.002*kernel,rtol=1e-6,atol=1e-9)
    assert not build_baseline().losses

def test_dropout_is_stochastic_only_during_training_and_saved_inference_matches(tmp_path):
    tf.keras.utils.set_random_seed(42);model=build_regularized_model_2();x=np.ones((10,30),dtype='float32')
    np.testing.assert_array_equal(model(x,training=False),model(x,training=False))
    assert not np.array_equal(model(x,training=True),model(x,training=True))
    path=tmp_path/'model.keras';model.save(path)
    loaded=tf.keras.models.load_model(path);np.testing.assert_array_equal(model(x,training=False),loaded(x,training=False))
