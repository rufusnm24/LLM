from pathlib import Path

from typer.testing import CliRunner

from cli.resume_opt import app


runner = CliRunner()


def test_cli_optimize_generates_artifacts(tmp_path):
    resume_path = Path("tests/data/sample_resume.tex")
    jd_path = Path("tests/data/sample_jd.txt")
    output_dir = tmp_path / "artifacts"

    result = runner.invoke(
        app,
        [
            "optimize",
            "--resume-path",
            str(resume_path),
            "--jd-file",
            str(jd_path),
            "--output-dir",
            str(output_dir),
            "--ats-threshold",
            "80",
            "--strict",
        ],
    )

    assert result.exit_code == 0, result.stdout
    optimized_tex = output_dir / "main_optimized.tex"
    diff_path = output_dir / "diff.patch"
    pdf_path = output_dir / "Resume_Optimized.pdf"
    report_path = output_dir / "report.md"

    assert optimized_tex.exists()
    assert diff_path.exists()
    assert pdf_path.exists()
    assert report_path.exists()

    report = report_path.read_text(encoding="utf-8")
    assert "After:" in report

    pdf_header = pdf_path.read_bytes()[:4]
    assert pdf_header == b"%PDF"
