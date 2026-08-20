"""Test the command-line scaffold."""

from idiolect import main


def test_main_has_no_output(capsys) -> None:
    """Check that the empty command writes no text."""
    main()

    assert capsys.readouterr().out == ""
