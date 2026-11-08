import pytest
import torch
from torch.nn import functional as F

from backend.core.data_structures import VariantKey
from backend.gnn.focal_loss import WeightedFocalLoss
from backend.gnn.graph_builder import (
    BiologicalNetwork, FEATURE_WIDTHS, REGULATES, RELATIONS, VariantRecord, build_graph,
)
from backend.gnn.model import DriverHGT


@pytest.fixture
def records():
    return [
        VariantRecord(VariantKey("22", 100, "A", "T"), [0.1] * 12, ("g1", "g1"), ("c1",)),
        VariantRecord(VariantKey("22", 200, "G", "C"), [0.3] * 12, ("g2",), ("c1",)),
        VariantRecord(VariantKey("22", 300, "A", "C"), [0.2] * 12),
    ]


@pytest.fixture
def network():
    return BiologicalNetwork({"g1": [1.0, 0.0], "g2": [0.0, 1.0]},
                             {"c1": [0.1, 0.3, 0.5, 0.7]}, (("c1", "g1"),), (("g1", "g2"),))


def test_graph_shapes_and_indices(records, network):
    graph = build_graph(records, network)
    assert graph.validate()
    assert set(graph.edge_types) == set(RELATIONS)
    for name, count in {"variant": 3, "gene": 2, "cell": 1}.items():
        assert graph[name].x.shape == (count, FEATURE_WIDTHS[name])
    torch.testing.assert_close(graph[REGULATES].edge_index, torch.tensor([[0, 1], [0, 1]]))
    with pytest.raises(ValueError, match="Duplicate"):
        build_graph([records[0], records[0]], network)
    with pytest.raises(ValueError, match="unknown"):
        build_graph(records, BiologicalNetwork({}, {}))
    with pytest.raises(ValueError, match="width"):
        build_graph(records, BiologicalNetwork({"g1": [0]}, {}))
    with pytest.raises(ValueError, match="at least one"):
        build_graph([], network)


def test_hgt_propagates_and_preserves_isolated_nodes(records, network):
    torch.manual_seed(19)
    model = DriverHGT().eval()
    graph = build_graph(records, network)
    original = model(graph)
    assert original.shape == (3,)
    graph["gene"].x[0] += 3
    changed = model(graph)
    assert abs((original - changed)[0].item()) > 1e-4
    torch.testing.assert_close(original[2], changed[2])
    empty = build_graph([records[-1]], BiologicalNetwork({}, {}))
    torch.testing.assert_close(model(empty)[0], original[2])
    partial = build_graph([records[-1], records[0]], BiologicalNetwork({"g1": [0, 1]}, {"c1": [0] * 4}))
    assert model(partial).shape == (2,)


def test_model_gradients_and_optimizer_step(records, network):
    torch.manual_seed(27)
    model = DriverHGT(dropout=0)
    graph = build_graph(records, network)
    graph["gene"].x.requires_grad_()
    loss_fn = WeightedFocalLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    targets = torch.tensor([1.0, 0.0, 0.0])
    loss = loss_fn(model(graph), targets)
    loss.backward()
    assert graph["gene"].x.grad.abs().sum() > 0
    attention_grads = [p.grad for p in model.layers.parameters() if p.grad is not None]
    assert attention_grads and all(torch.isfinite(grad).all() for grad in attention_grads)
    assert any(grad.abs().sum() > 0 for grad in attention_grads)
    optimizer.step()
    assert loss_fn(model(graph), targets).item() < loss.item()


def test_focal_loss_weighting_reductions_and_extreme_logits():
    logits = torch.tensor([1000.0, -1000.0, 0.0, 0.0], requires_grad=True)
    targets = torch.tensor([1, 0, 1, 0])
    per_example = WeightedFocalLoss(alpha=0.75, gamma=0, reduction="none")(logits, targets)
    expected = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
    torch.testing.assert_close(per_example, expected * torch.tensor([0.75, 0.25, 0.75, 0.25]))
    focused = WeightedFocalLoss(gamma=0.5, reduction="sum")(logits, targets)
    assert focused < per_example.sum()
    focused.backward()
    assert torch.isfinite(logits.grad).all()
    with pytest.raises(ValueError, match="binary"):
        WeightedFocalLoss()(logits, torch.full_like(logits, 0.5))
    with pytest.raises(ValueError, match="same"):
        WeightedFocalLoss()(logits, targets[:1])
    with pytest.raises(ValueError, match="gamma"):
        WeightedFocalLoss(gamma=-1)
