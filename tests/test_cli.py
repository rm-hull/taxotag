from __future__ import annotations

import json

from taxotag import Topic, cli


class FakeGist:
    def __init__(self, **_: object) -> None:
        pass

    def classify(
        self,
        text: str,
        *,
        top_k: int,
        threshold: float | None,
    ) -> list[Topic]:
        assert text == "example text"
        assert top_k == 2
        assert threshold == 0.75
        return [Topic("technology", "Technology & Software", 0.9)]


def test_json_output(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "Gist", FakeGist)

    assert (
        cli.main(
            [
                "example text",
                "--top-k",
                "2",
                "--threshold",
                "0.75",
                "--json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == [
        {"slug": "technology", "name": "Technology & Software", "score": 0.9}
    ]


def test_text_output(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "Gist", FakeGist)

    assert cli.main(["example text", "--top-k", "2", "--threshold", "0.75"]) == 0
    assert capsys.readouterr().out == "technology\tTechnology & Software\t0.900000\n"
