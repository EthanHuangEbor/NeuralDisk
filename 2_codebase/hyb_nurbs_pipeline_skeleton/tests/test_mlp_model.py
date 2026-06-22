from hyb_nurbs.ml.model import MLPRegressor


def test_mlp_forward_output_dimension():
    import torch

    model = MLPRegressor(input_dim=24, output_dim=30, hidden_dims=[8, 8], activation="relu")
    y = model(torch.zeros(2, 24))
    assert tuple(y.shape) == (2, 30)
