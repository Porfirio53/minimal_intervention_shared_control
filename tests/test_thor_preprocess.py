from pathlib import Path

from minimal_intervention_shared_control.datasets.thor import convert_thor_tsv


def test_thor_conversion_centers_markers_and_segments_missing_frames(
    tmp_path: Path,
) -> None:
    source = tmp_path / "run.tsv"
    source.write_text(
        "NO_OF_FRAMES\t3\n"
        "FREQUENCY\t10\n"
        "DATA_INCLUDED\t3D\n"
        "Frame\tTime\tHelmet_1 - 1 X\tHelmet_1 - 1 Y\tHelmet_1 - 1 Z\t"
        "Helmet_1 - 2 X\tHelmet_1 - 2 Y\tHelmet_1 - 2 Z\n"
        "1\t0.0\t1000\t2000\t0\t3000\t4000\t0\n"
        "2\t0.1\t0\t0\t0\t0\t0\t0\n"
        "3\t0.3\t2000\t3000\t0\t4000\t5000\t0\n",
        encoding="utf-8",
    )
    target = tmp_path / "normalized.tsv"
    assert convert_thor_tsv(source, target) == 2
    assert target.read_text(encoding="utf-8").splitlines() == [
        "id\ttime\tx\ty",
        "run:Helmet_1:000\t0.0\t2.0\t3.0",
        "run:Helmet_1:001\t0.3\t3.0\t4.0",
    ]
