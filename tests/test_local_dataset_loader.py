from Orcar.load_cache_dataset import load_local_dataset
import pathlib
import csv


def test_load_local_dataset_minimal(tmp_path):
    headers = [
        "repo",
        "instance_id",
        "base_commit",
        "patch",
        "problem_statement",
        "version",
        "created_at",
    ]
    row = [
        "owner__repo",
        "owner__repo-1",
        "deadbeef",
        "# sample patch",
        "Sample bug description",
        "3.8",
        "2026-05-26T00:00:00Z",
    ]

    p = tmp_path / "test.csv"
    with open(p, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerow(row)

    ds = load_local_dataset(str(p), split="test")
    assert len(ds) == 1
    inst = ds[0]
    assert inst["instance_id"] == "owner__repo-1"
    assert inst.get("test_patch", "") == ""
    assert inst.get("hints_text", "") == ""
    assert inst.get("environment_setup_commit", inst["base_commit"]) == inst["base_commit"]

def test_load_local_dataset(tmp_path):
    """
    Test loading a local dataset with real data.
    """
    p = tmp_path / "data.csv"
    ds = load_local_dataset(str(p))
    assert len(ds) == 1
    inst = ds[0]
    print(inst["instance_id"])
    assert inst.get("environment_setup_commit", inst["base_commit"]) == inst["base_commit"]

if __name__ == "__main__":
    test_load_local_dataset_minimal(pathlib.Path("data/"))
    test_load_local_dataset(pathlib.Path("data/"))