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


def test_datasets_keeps_only_the_chosen_groups(tmp_path):
    run = tmp_path / "run"
    write_csv(run / "metrics_test.csv", [{"case": "x", "dice_anterior": 0.6, "dice_posterior": 0.8,
                                          "nsd_anterior": 0.9, "nsd_posterior": 0.7}])
    write_csv(run / "external" / "metrics_external.csv", [{"case": "s01_L", "dice": 0.8, "nsd": 0.75}])
    write_csv(run / "shifted_external" / "z-5" / "metrics_external.csv", [{"case": "s01_L", "dice": 0.9, "nsd": 0.9}])

    rows = metrics_plot.main([str(run)], str(tmp_path / "fig"), datasets=["internal", "external"])
    assert {r["dataset"] for r in rows} == {"internal", "external"}          # shifted group left out
    assert (tmp_path / "fig.png").stat().st_size > 0
    fig = metrics_plot.plot({"run": {d: v for d, v in metrics_plot.case_scores(run).items() if d != "external_shifted"}})
    assert [t.get_text() for t in fig.legends[0].get_texts()] == [metrics_plot.DATASETS[0][1], metrics_plot.DATASETS[1][1]]


def test_external_largest_group(tmp_path):
    run = tmp_path / "run"
    write_csv(run / "external" / "metrics_external.csv", [{"case": "s01_L", "dice": 0.8, "nsd": 0.75}])
    write_csv(run / "external_largest" / "metrics_external.csv", [{"case": "s01_L", "dice": 0.85, "nsd": 0.8}])
    rows = metrics_plot.main([str(run)], str(tmp_path / "fig"), datasets=["external", "external_largest"])
    got = {(r["dataset"], r["metric"]): r["mean"] for r in rows}
    assert got[("external_largest", "dice")] == 0.85 and got[("external", "dice")] == 0.8
    fig = metrics_plot.plot({"run": metrics_plot.case_scores(run)})
    assert [t.get_text() for t in fig.legends[0].get_texts()] == [metrics_plot.DATASETS[1][1], metrics_plot.DATASETS[2][1]]


def test_names_label_the_runs(tmp_path):
    run = tmp_path / "baseline"
    write_csv(run / "external" / "metrics_external.csv", [{"case": "s01_L", "dice": 0.8, "nsd": 0.75}])
    rows = metrics_plot.main([str(run)], str(tmp_path / "fig"), names=["3D U-Net"])
    assert {r["run"] for r in rows} == {"3D U-Net"}
    assert "3D U-Net" in (tmp_path / "fig.csv").read_text() and "baseline" not in (tmp_path / "fig.csv").read_text()


def test_unscored_cases_left_out(tmp_path):
    run = tmp_path / "run"
    blank = {"dice_anterior": "", "dice_posterior": "", "nsd_anterior": "", "nsd_posterior": ""}
    write_csv(run / "metrics_test.csv", [{"case": "x", "dice_anterior": 0.6, "dice_posterior": 0.8,
                                          "nsd_anterior": 0.9, "nsd_posterior": 0.7}, {"case": "y", **blank}])
    scores = metrics_plot.case_scores(run)
    assert list(scores["internal"]["dice"]) == [0.7]


def test_shifted_group_averages_each_case_over_shifts(tmp_path):
    run = tmp_path / "run"
    row = lambda c, d: {"case": c, "dice_anterior": d, "dice_posterior": d, "nsd_anterior": d, "nsd_posterior": d}
    write_csv(run / "shifted" / "none" / "metrics_test.csv", [row("a", 0.0), row("b", 0.0)])   # control: left out
    write_csv(run / "shifted" / "x+5" / "metrics_test.csv", [row("a", 0.8), row("b", 0.6)])
    write_csv(run / "shifted" / "x-5" / "metrics_test.csv", [row("a", 0.6), row("b", 0.4)])

    scores = metrics_plot.case_scores(run)
    assert list(scores) == ["shifted"]                              # no other datasets yet
    assert list(scores["shifted"]["dice"].round(6)) == [0.7, 0.5]   # mean over x+5 and x-5, per case
    rows = metrics_plot.main([str(run)], str(tmp_path / "fig"))
    assert {(r["dataset"], r["n"]) for r in rows} == {("shifted", 2)}


def test_external_shifted_group(tmp_path):
    run = tmp_path / "run"
    row = lambda c, d: {"case": c, "dice": d, "nsd": d, "label_coverage": 1.0}
    write_csv(run / "external" / "metrics_external.csv", [row("s01_L", 0.8)])
    write_csv(run / "shifted_external" / "none" / "metrics_external.csv", [row("s01_L", 0.8)])  # control: left out
    write_csv(run / "shifted_external" / "y+5" / "metrics_external.csv", [row("s01_L", 0.7)])
    write_csv(run / "shifted_external" / "y-5" / "metrics_external.csv", [row("s01_L", 0.5)])
    scores = metrics_plot.case_scores(run)
    assert list(scores) == ["external", "external_shifted"]
    assert list(scores["external_shifted"]["dice"].round(6)) == [0.6]
    metrics_plot.main([str(run)], str(tmp_path / "fig"))
