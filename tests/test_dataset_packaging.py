"""Dataset packaging tests: catdog corpus integrity, toxicity split/verify tools.

The toxicity split CSVs contain real third-party social-media texts and are NOT
shipped; the toxicity tests here use a synthetic labeled corpus with the same
schema and prove the split builder + manifest verifier round-trip works.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
DATASETS = REPO / "datasets"
TOX = DATASETS / "toxicity"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, TOX / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def prepare():
    return _load("prepare_toxicity_splits")


@pytest.fixture(scope="module")
def verify():
    return _load("verify_split")


class TestCatdogCorpus:
    def test_corpus_shipped_with_expected_schema(self):
        path = DATASETS / "catdog" / "corpus.csv"
        assert path.exists(), "corpus.csv must be shipped"
        frame = pd.read_csv(path)
        assert len(frame) == 1041
        expected = {
            "text",
            "concept",
            "is_dog",
            "style",
            "style_bin",
            "topic",
            "topic_bin",
            "language",
            "language_bin",
            "candidate_id",
            "source_text",
        }
        assert set(frame.columns) == expected
        assert set(frame["concept"].unique()) == {"cat", "dog"}
        assert set(frame["style"].unique()) == {"shakespeare", "preschool"}
        assert set(frame["topic"].unique()) == {"horror", "sci_fi"}
        assert set(frame["language"].unique()) == {"en", "ar"}
        assert set(frame["is_dog"].unique()) <= {0, 1}
        assert frame["candidate_id"].is_unique

    def test_dataset_doc_documents_models_and_license(self):
        text = (DATASETS / "catdog" / "DATASET.md").read_text(encoding="utf-8")
        for needle in (
            "Mistral-7B-Instruct-v0.2",
            "Hy-MT2-1.8B",
            "gemma-4",
            "MIT",
            "shakespeare",
            "preschool",
            "horror",
            "sci_fi",
        ):
            assert needle in text


class TestToxicityPackaging:
    def test_manifest_structure(self):
        manifest = json.loads((TOX / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["dataset"] == "toxicity"
        assert manifest["split_seed"] == 42
        assert manifest["num_candidates"] == 856
        assert manifest["split_counts"] == {
            "geometry_400": 400,
            "vector_construction_256": 256,
            "evaluation_200": 200,
        }
        assert len(manifest["candidates"]) == 856
        for entry in manifest["candidates"].values():
            assert entry["split"] in {
                "geometry_400",
                "vector_construction_256",
                "evaluation_200",
            }
            assert entry["cell"][0] in ("en", "ar")
            assert len(entry["text_sha256"]) == 64

    def test_no_texts_shipped(self):
        shipped = [p.name for p in TOX.iterdir()]
        for name in shipped:
            assert not name.endswith(".csv"), f"texts must not ship: {name}"

    def test_split_build_and_verify_round_trip(self, prepare, verify, tmp_path):
        rng = __import__("random").Random(0)
        # Synthetic labeled corpus with the exact schema of {en,ar}_labeled.csv;
        # overfill each (language, toxic, gender, location-ok) pool so the
        # splitter can draw 50/32/25 per (language, toxicity, gender) cell.
        rows = []
        for language in ("en", "ar"):
            pool = []
            for toxic in (0, 1):
                for gender in (0, 1):
                    for index in range(120):
                        pool.append(
                            {
                                "text": f"{language} post {toxic} {gender} {index} "
                                + rng.choice(["sad", "great", "awful", "nice"]),
                                "toxic": toxic,
                                "mentions_gender": gender,
                                "mentions_location": 0,
                            }
                        )
            frame = pd.DataFrame(pool).drop_duplicates("text")
            frame.to_csv(tmp_path / f"{language}_labeled.csv", index=False)
            rows.append(frame)
        split_dir = tmp_path / "splits"
        prepare.build_splits(str(tmp_path), str(split_dir))
        sizes = {"geometry_400": 400, "vector_construction_256": 256, "evaluation_200": 200}
        per_cell = {"geometry_400": 50, "vector_construction_256": 32, "evaluation_200": 25}
        for name, expected in sizes.items():
            frame = pd.read_csv(split_dir / f"{name}.csv")
            assert len(frame) == expected
            assert frame.groupby(["language", "toxicity_bin", "gender_bin"]).size().eq(
                per_cell[name]
            ).all()

        # Build a manifest from the synthetic splits the same way the shipped
        # one was built, then verify.
        manifest = {
            "manifest_version": 1,
            "dataset": "toxicity",
            "split_seed": 42,
            "candidates": {},
        }
        for name in sizes:
            frame = pd.read_csv(split_dir / f"{name}.csv")
            for _, row in frame.iterrows():
                text = row["text"].strip()
                cid = row["candidate_id"]
                manifest["candidates"][cid] = {
                    "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "cell": [row["language"], str(row["toxicity_bin"]), str(row["gender_bin"])],
                    "split": name,
                }
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        exit_code = verify.verify_splits(str(split_dir), str(manifest_path))
        assert exit_code == 0

    def test_verify_detects_mismatch(self, verify, tmp_path):
        manifest = {
            "candidates": {
                "deadbeefdeadbeef": {
                    "text_sha256": "0" * 64,
                    "cell": ["en", "0", "0"],
                    "split": "geometry_400",
                }
            }
        }
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        split_dir = tmp_path / "splits"
        split_dir.mkdir()
        frame = pd.DataFrame(
            [
                {
                    "text": "hello world",
                    "toxicity": 0,
                    "toxicity_bin": 0,
                    "gender": 0,
                    "gender_bin": 0,
                    "language": "en",
                    "language_bin": 0,
                    "candidate_id": hashlib.sha256(b"hello world").hexdigest()[:16],
                }
            ]
        )
        frame.to_csv(split_dir / "geometry_400.csv", index=False)
        assert verify.verify_splits(str(split_dir), str(manifest_path)) == 1

    def test_dataset_doc_documents_provenance_and_prompt(self):
        text = (TOX / "DATASET.md").read_text(encoding="utf-8")
        for needle in (
            "textdetox/multilingual_toxicity_dataset",
            "openrail++",
            "Jigsaw",
            "OSACT4",
            "gpt-4.1-mini",
            "You are labeling short social media texts.",
            "mentions_location",
        ):
            assert needle in text
