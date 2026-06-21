"""Regression tests for lightweight ML-XC calculator plumbing."""

import numpy as np

from atom.xc.ml_xc import MLXCCalculator, XCBaseModel


class DummyNetwork:
    pass


class DummyXCModel(XCBaseModel):
    def load_model(self, model_dir, model_name="xc"):
        raise NotImplementedError

    def save_model(self, model_dir, model_name="xc"):
        raise NotImplementedError

    def eval_model(self, data_loader):
        raise NotImplementedError

    def train(self, train_loader, val_loader=None, **kwargs):
        raise NotImplementedError

    def forward(self, features):
        return np.sum(features, axis=1)


def test_ml_xc_model_from_class_without_model_dir_predicts():
    model = DummyXCModel(
        model_kind="potential",
        features_list=["rho"],
        weights_ext="pt",
        config_ext="json",
        model_cls=DummyNetwork,
        model_init_kwargs={},
        device="cpu",
    )
    calc = MLXCCalculator(
        model=model,
        features_list=["rho"],
        target_functional="GGA_PBE",
        target_component="v_xc",
        target_mode="delta",
        reference_functional="LDA_PW",
        scale_features=False,
        scale_targets=False,
        scaler_type_features="none",
        scaler_type_targets="none",
        use_symlog_features=False,
        use_symlog_targets=False,
    )

    predictions = calc.predict_vxc(np.array([[1.0], [2.0]]))

    assert model.model_dir is None
    np.testing.assert_allclose(predictions, np.array([1.0, 2.0]))
