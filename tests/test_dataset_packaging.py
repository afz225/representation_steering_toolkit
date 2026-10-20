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
        assert manifest["manifest_version"] >= 2
        for split, size in manifest["split_counts"].items():
            rows = sorted(e["row"] for e in manifest["candidates"].values() if e["split"] == split)
            assert rows == list(range(size))

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


def _synthetic_labeled(tmp_path, per_cell=120):
    """{en,ar}_labeled.csv with the production schema, overfilled per cell."""
    rng = __import__("random").Random(0)
    for language in ("en", "ar"):
        pool = [
            {
                "text": f"{language} post {toxic} {gender} {index} " + rng.choice(["sad", "great", "awful"]),
                "toxic": toxic,
                "mentions_gender": gender,
                "mentions_location": 0,
            }
            for toxic in (0, 1)
            for gender in (0, 1)
            for index in range(per_cell)
        ]
        pd.DataFrame(pool).to_csv(tmp_path / f"{language}_labeled.csv", index=False)


def _manifest_from_splits(split_dir, path):
    manifest = {"manifest_version": 2, "dataset": "toxicity", "split_seed": 42, "candidates": {}}
    for name in ("geometry_400", "vector_construction_256", "evaluation_200"):
        frame = pd.read_csv(split_dir / f"{name}.csv")
        for position, row in frame.iterrows():
            manifest["candidates"][row["candidate_id"]] = {
                "text_sha256": hashlib.sha256(row["text"].strip().encode("utf-8")).hexdigest(),
                "cell": [row["language"], str(row["toxicity_bin"]), str(row["gender_bin"])],
                "split": name,
                "row": int(position),
            }
    path.write_text(json.dumps(manifest), encoding="utf-8")


class TestRebuildFromManifest:
    @pytest.fixture()
    def built(self, prepare, tmp_path):
        _synthetic_labeled(tmp_path)
        split_dir = tmp_path / "splits"
        prepare.build_splits(str(tmp_path), str(split_dir))
        _manifest_from_splits(split_dir, tmp_path / "manifest.json")
        return split_dir

    def _source(self, tmp_path, pad=True):
        source = tmp_path / "source"
        source.mkdir()
        for language in ("en", "ar"):
            frame = pd.read_csv(tmp_path / f"{language}_labeled.csv")
            # Source texts as a user downloads them: surrounding whitespace, extra
            # unused rows, original column layout.
            text = ("  " + frame["text"] + " \n") if pad else frame["text"]
            pd.DataFrame({"text": text, "toxic": frame["toxic"]}).to_csv(source / f"{language}.csv", index=False)
        return source

    def test_rebuild_is_byte_identical_and_verifies(self, verify, built, tmp_path):
        rebuild = _load("rebuild_from_manifest")
        out = tmp_path / "rebuilt"
        assert rebuild.rebuild(str(out), str(tmp_path / "manifest.json"), str(self._source(tmp_path))) == 0
        for name in ("geometry_400", "vector_construction_256", "evaluation_200"):
            assert (out / f"{name}.csv").read_bytes() == (built / f"{name}.csv").read_bytes()
        assert verify.verify_splits(str(out), str(tmp_path / "manifest.json")) == 0

    def test_rebuild_reports_missing_texts(self, built, tmp_path):
        rebuild = _load("rebuild_from_manifest")
        source = self._source(tmp_path, pad=False)
        frame = pd.read_csv(source / "ar.csv")
        frame.iloc[: len(frame) // 2].to_csv(source / "ar.csv", index=False)
        assert rebuild.rebuild(str(tmp_path / "out"), str(tmp_path / "manifest.json"), str(source)) == 1

    def test_verify_checks_row_order(self, verify, built, tmp_path):
        frame = pd.read_csv(built / "evaluation_200.csv")
        frame.iloc[::-1].to_csv(built / "evaluation_200.csv", index=False)
        manifest = str(tmp_path / "manifest.json")
        assert verify.verify_splits(str(built), manifest) == 1
        assert verify.verify_splits(str(built), manifest, check_order=False) == 0


class TestBatchLabeler:
    @pytest.fixture(scope="class")
    def labeler(self):
        spec = importlib.util.spec_from_file_location(
            "batch_label_toxicity", REPO / "scripts" / "toxicity" / "batch_label_toxicity.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def _output_line(custom_id, labels, parts=1):
        content = [{"type": "output_text", "text": ""}] * (parts - 1) + [
            {"type": "output_text", "text": json.dumps(labels)}
        ]
        return {
            "id": "batch_req_x",
            "custom_id": custom_id,
            "response": {"status_code": 200, "body": {"output": [{"type": "message", "content": content}]}},
            "error": None,
        }

    def test_build_batch_uses_raw_text_and_production_ids(self, labeler, tmp_path):
        texts = tmp_path / "texts.csv"
        pd.DataFrame({"text": [" padded text ", "plain"], "toxic": [0, 1]}).to_csv(texts, index=False)
        args = __import__("argparse").Namespace(input=str(texts), language="en", output_dir=str(tmp_path / "b"), max_rows=1)
        labeler.cmd_build_batch(args)
        lines = [json.loads(p.read_text()) for p in sorted((tmp_path / "b").glob("*.jsonl"))]
        assert [line["custom_id"] for line in lines] == ["en_000000", "en_000001"]
        assert lines[0]["body"]["input"][1]["content"][0]["text"] == "Text:  padded text "
        assert lines[0]["url"] == "/v1/responses" and lines[0]["body"]["model"] == "gpt-4.1-mini"

    def test_assemble_matches_by_custom_id(self, labeler, tmp_path):
        texts = tmp_path / "texts.csv"
        pd.DataFrame({"text": ["a", "b", "c"], "toxic": [0, 1, 0]}).to_csv(texts, index=False)
        labels = {f: (1 if f.startswith("mentions_") else "x") for f in labeler.LABEL_FIELDS}
        records = [
            self._output_line("en_000002", {**labels, "mentions_gender": 0}),  # out of order
            self._output_line("en_000000", labels, parts=2),  # empty first part, as in production
            {"custom_id": "en_000001", "response": {"status_code": 500, "body": {}}, "error": {"code": "x"}},
        ]
        results = tmp_path / "out_output.jsonl"
        results.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
        ns = __import__("argparse").Namespace
        out = tmp_path / "en_labeled.csv"
        with pytest.raises(SystemExit):
            labeler.cmd_assemble(ns(input=str(texts), language="en", results=str(results), output=str(out), allow_missing=False))
        labeler.cmd_assemble(ns(input=str(texts), language="en", results=str(results), output=str(out), allow_missing=True))
        frame = pd.read_csv(out, dtype=str, keep_default_na=False)
        assert list(frame.columns) == labeler.LABELED_COLUMNS
        assert frame.loc[0, "mentions_gender"] == "1" and frame.loc[2, "mentions_gender"] == "0"
        assert frame.loc[1, "mentions_gender"] == "" and frame.loc[1, "toxic"] == "1"
