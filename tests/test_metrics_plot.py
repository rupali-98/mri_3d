"""hippo.metrics_plot reads both datasets' per-case metrics, plots them and writes a summary table."""
import csv

from hippo import metrics_plot


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_plot_and_summary(tmp_path):
    a, b = tmp_path / "run_a", tmp_path / "run_b"
    write_csv(a / "metrics_test.csv", [{"case": c, "dice_anterior": d, "dice_posterior": d + 0.2,
                                        "nsd_anterior": 0.9, "nsd_posterior": 0.7} for c, d in [("x", 0.6), ("y", 0.7)]])
    write_csv(a / "external" / "metrics_external.csv", [{"case": "s01_L", "dice": 0.8, "nsd": 0.75}])
    write_csv(b / "metrics_test.csv", [{"case": "x", "dice_anterior": 0.5, "dice_posterior": 0.5,
                                        "nsd_anterior": 0.5, "nsd_posterior": 0.5}])   # no external yet

    rows = metrics_plot.main([str(a), str(b)], str(tmp_path / "fig"))
    got = {(r["run"], r["dataset"], r["metric"]): r for r in rows}
    assert got[("run_a", "internal", "dice")]["mean"] == 0.75        # mean of parts per case: 0.7, 0.8
    assert got[("run_a", "internal", "nsd")]["n"] == 2
    assert got[("run_a", "external", "dice")]["mean"] == 0.8
    assert ("run_b", "external", "dice") not in got
    assert (tmp_path / "fig.png").stat().st_size > 0
    assert (tmp_path / "fig.csv").read_text().startswith("run,dataset,metric,n,mean,median,std")
