from argparse import Namespace
from dataclasses import replace
from itertools import product

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("decodanda")

from repsteer import cli  # noqa: E402
from repsteer.config import DatasetConfig, GeometryConfig, ModelConfig  # noqa: E402


class FakeBackend:
    """Deterministic per-layer representations with a layer-dependent signal."""

    def sentence_representations_layers(self, texts, layers, pooling, batch_size, max_length):
        out = {}
        for layer in layers:
            rng = np.random.default_rng(layer)
            codes = np.array([[int(c) for c in t.split("|")[0]] for t in texts], dtype=float)
            out[layer] = (rng.normal(size=(len(texts), 24))
                          + (layer + 1) * 0.4 * codes @ rng.normal(size=(3, 24))).astype(np.float32)
        return out


def _frame(n_per_cell=12):
    rows = []
    for cell in product("01", repeat=3):
        for i in range(n_per_cell):
            rows.append({"id": f"{''.join(cell)}-{i}", "text": f"{''.join(cell)}|{i}",
                         "a": cell[0], "b": cell[1], "c": cell[2]})
    return pd.DataFrame(rows)


def _config(protocol):
    geometry = GeometryConfig(layers=(2, 5, 9), protocol=protocol, n_per_cell=10, audit_n_per_cell=10,
                              ndata=6, sd_ndata=4, n_resamplings=3, sd_cross_validations=3,
                              audit_cross_validations=3, variables=("a", "b", "c"))
    dataset = DatasetConfig(path="unused.csv", text_column="text", id_column="id",
                            variables={"a": ("0", "1"), "b": ("0", "1"), "c": ("0", "1")})
    return Namespace(geometry=geometry, dataset=dataset, model=ModelConfig(model_id="fake"))


@pytest.mark.parametrize("protocol, outputs", [
    ("production", ("geometry_rows.csv", "dichotomy_rows.csv")),
    ("matched_audit", ("matched_metrics.csv",)),
])
def test_layer_workers_match_sequential(tmp_path, protocol, outputs):
    config = _config(protocol)
    frames = {}
    for workers in (1, 3):
        out = tmp_path / f"w{workers}"
        out.mkdir()
        cli._decodanda_geometry(config, FakeBackend(), _frame(), Namespace(dataset="d.csv", layer_workers=workers),
                                out, relative_depth_of=lambda layer: layer / 10)
        frames[workers] = {name: pd.read_csv(out / name).drop(columns=["elapsed_sec"], errors="ignore")
                           for name in outputs}
    for name in outputs:
        sequential, parallel = frames[1][name], frames[3][name]
        timing = [c for c in sequential.columns if c.endswith("_time_sec")]
        pd.testing.assert_frame_equal(sequential.drop(columns=timing), parallel.drop(columns=timing))
        assert sorted(sequential["layer"].unique()) == [2, 5, 9]
