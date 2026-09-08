import ast
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "06_fm_before_classifier.ipynb"


def notebook_cell(index):
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "".join(payload["cells"][index].get("source", []))


def stage3_namespace():
    namespace = {
        "copy": copy,
        "json": json,
        "random": random,
        "time": time,
        "Path": Path,
        "np": np,
        "pd": pd,
        "torch": torch,
        "nn": nn,
        "F": F,
        "DEVICE": torch.device("cpu"),
        "S3": {
            "hidden_dim": 16,
            "num_hidden_layers": 2,
            "zero_init_output": True,
            "learning_rate": 1e-2,
            "head_learning_rate": 1e-2,
            "weight_decay": 0.0,
            "batch_size": 12,
            "max_epochs": 3,
            "early_stopping_patience": 5,
            "lr_patience": 5,
            "min_delta": -1.0,
            "checkpoint_interval": 10,
            "resume_training": False,
            "lambda_displacement": 0.0,
            "lambda_velocity": 0.0,
            "guidance_step_size": 0.1,
            "guidance_num_steps": 2,
            "guidance_normalize": "trust_region",
            "guidance_anchor": "source",
            "guidance_monotone": True,
            "target_refresh_every": 1,
            "unfreeze_epoch": 1,
        },
    }

    def seed_everything(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    @torch.no_grad()
    def evaluate_system(net, head, z, y, T, chunk=4096):
        predictions = []
        total_loss = 0.0
        correct = 0
        for start in range(0, len(z), chunk):
            zb, yb = z[start : start + chunk], y[start : start + chunk]
            z_hat = zb if net is None else namespace["euler_rollout"](net, zb, T)
            logits = head(z_hat)
            total_loss += F.cross_entropy(logits, yb, reduction="sum").item()
            pred = logits.argmax(1)
            predictions.append(pred)
            correct += int((pred == yb).sum())
        return correct / len(z), total_loss / len(z), torch.cat(predictions)

    def head_anchor_penalty(head, w0, b0):
        return (head.weight - w0).pow(2).sum() + (head.bias - b0).pow(2).sum()

    namespace.update(
        seed_everything=seed_everything,
        evaluate_system=evaluate_system,
        head_anchor_penalty=head_anchor_penalty,
    )
    for index in (12, 16, 18):
        exec(compile(notebook_cell(index), f"cell-{index}", "exec"), namespace)

    # Cell 20 ends by running a guard against real Stage 1 globals. Execute its definitions only.
    tree = ast.parse(notebook_cell(20))
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "verify_stage3_core_behaviour"
    )]
    exec(compile(tree, "cell-20", "exec"), namespace)
    return namespace


def synthetic_problem(n=48, d=6):
    generator = torch.Generator().manual_seed(7)
    z = torch.randn(n, d, generator=generator)
    y = (z[:, 0] + 0.35 * z[:, 1] > 0).long()
    return z[:24], y[:24], z[24:36], y[24:36], z[36:], y[36:]


def test_every_python_cell_parses():
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for index, cell in enumerate(payload["cells"]):
        if cell.get("cell_type") != "code":
            continue
        text = "".join(cell.get("source", []))
        if text.lstrip().startswith("%"):
            continue
        ast.parse(text, filename=f"cell-{index}")


def test_guided_targets_are_source_bounded_and_monotone():
    ns = stage3_namespace()
    z_train, y_train, *_ = synthetic_problem()
    head = nn.Linear(z_train.shape[1], 2)
    head.requires_grad_(False)
    net = ns["make_velocity_net"](z_train.shape[1])
    scale = z_train.norm(dim=1).mean().item()
    targets, diag = ns["classifier_guided_targets"](
        net, head, z_train, y_train, 4, 0.1, 3, "trust_region", scale,
        anchor_mode="source", monotone=True,
    )
    assert (targets - z_train).norm(dim=1).max().item() <= 0.1 * scale * (1 + 1e-5)
    assert diag["guidance_target_ce_after"] <= diag["guidance_target_ce_before"] + 1e-6
    assert 0.0 <= diag["guidance_trust_region_hit_rate"] <= 1.0


def test_relative_displacement_penalty_is_scale_invariant():
    ns = stage3_namespace()

    class ScaleVelocity(nn.Module):
        def forward(self, z, t):
            return 0.25 * z

    head = nn.Linear(5, 2)
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)
    head.requires_grad_(False)
    z = torch.randn(10, 5)
    y = torch.arange(10) % 2
    loss_a, _ = ns["strategy1_batch_loss"](ScaleVelocity(), head, z, y, 4, 2.0, 0.0)
    loss_b, _ = ns["strategy1_batch_loss"](ScaleVelocity(), head, 7 * z, y, 4, 2.0, 0.0)
    assert torch.allclose(loss_a, loss_b, atol=1e-6)


def test_guided_joint_training_updates_the_classifier(tmp_path):
    ns = stage3_namespace()
    z_train, y_train, z_val, y_val, *_ = synthetic_problem()
    frozen_head = nn.Linear(z_train.shape[1], 2)
    frozen_head.requires_grad_(False)
    joint_head = copy.deepcopy(frozen_head)
    initial = {name: value.detach().clone() for name, value in joint_head.state_dict().items()}
    run_config = {"test": "guided-joint", "revision": 2}
    ns["train_stage3"](
        "guided", z_train, y_train, z_val, y_val, frozen_head, 4, tmp_path,
        run_config, init_seed=0, joint_head=joint_head, verbose=False,
    )
    assert any(not torch.equal(initial[name], value) for name, value in joint_head.state_dict().items())
    assert all(parameter.grad is None for parameter in frozen_head.parameters())


def test_cache_config_covers_implementation_and_upstream_artifacts():
    text = notebook_cell(22)
    for required in (
        "implementation_revision",
        "upstream_signature",
        "max_epochs",
        "early_stopping_patience",
        "effective_hyper",
    ):
        assert required in text


if __name__ == "__main__":
    import tempfile

    test_every_python_cell_parses()
    test_guided_targets_are_source_bounded_and_monotone()
    test_relative_displacement_penalty_is_scale_invariant()
    with tempfile.TemporaryDirectory() as directory:
        test_guided_joint_training_updates_the_classifier(Path(directory))
    test_cache_config_covers_implementation_and_upstream_artifacts()
    print("5 Stage 3 checks passed")
