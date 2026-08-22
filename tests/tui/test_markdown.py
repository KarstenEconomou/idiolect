"""Test focused Markdown rendering for chat messages."""

from rich.console import Console
from rich.segment import Segment
from rich.style import Style

from idiolect.tui.markdown import ChatMarkdown


def test_formats_supported_inline_text_and_retains_heading_prefixes() -> None:
    """Check focused inline styles, headings, and literal unsupported markup."""
    source = (
        "**bold** *italic* ***both*** `code`\n\n"
        "## Head\n\n"
        "[mail](mailto:user@example.test) <u>underline</u> "
        "[bold]literal[/bold]\n\n"
        "unmatched **marker"
    )

    segments = _segments(source)

    assert _plain(segments) == (
        "bold italic both code\n\n"
        "## Head\n\n"
        "[mail](mailto:user@example.test) <u>underline</u> "
        "[bold]literal[/bold]\n\n"
        "unmatched **marker\n"
    )
    assert _style(segments, "## ").bold
    assert _style(segments, "Head").bold
    assert _style(segments, "bold").bold
    assert _style(segments, "italic").italic
    assert _style(segments, "both").bold
    assert _style(segments, "both").italic
    code_style = _style(segments, "code")
    assert code_style.color is None
    assert code_style.bgcolor is not None
    assert code_style.bgcolor.number == 8
    assert not code_style.dim
    underline_style = _segment_containing(segments, "<u>underline</u>").style
    assert underline_style is None or underline_style.underline is not True


def test_indents_each_list_level_and_aligns_wrapped_item_text() -> None:
    """Check authored list markers, nesting, numbering, and hanging wraps."""
    source = (
        "- alpha beta gamma delta\n"
        "  + nested\n\n"
        "3) third\n"
        "7) seventh"
    )

    assert _capture(source, width=16) == (
        " - alpha beta \n"
        "   gamma delta\n"
        "  + nested\n"
        "\n"
        " 3) third\n"
        " 7) seventh\n"
    )


def test_indents_each_quote_level_and_aligns_wrapped_quote_text() -> None:
    """Check retained quote markers, nesting, emphasis, and hanging wraps."""
    source = (
        "> alpha beta gamma delta\n"
        "> > nested\n"
        "> **bold**\n"
        "`> inline code`\n"
        "\\> escaped"
    )

    segments = _segments(source, width=16)

    assert _plain(segments) == (
        " > alpha beta \n"
        "   gamma delta\n"
        "  >> nested\n"
        " > bold\n"
        "> inline code\n"
        "> escaped\n"
    )
    assert _style(segments, "bold").bold


def test_formats_safe_links_with_visible_muted_destinations() -> None:
    """Check link styling, visible targets, and literal unsupported links."""
    source = (
        "Read [the **docs**](https://example.test/guide?q=chat) or "
        "https://bare.test.\n"
        "[mail](mailto:user@example.test) [broken](https://example.test"
    )

    segments = _segments(source)

    assert _plain(segments) == (
        "Read the docs (https://example.test/guide?q=chat) or "
        "https://bare.test.\n"
        "[mail](mailto:user@example.test) [broken](https://example.test\n"
    )
    link_text = _style(segments, "the ")
    assert not link_text.dim
    assert link_text.underline
    assert link_text.link == "https://example.test/guide?q=chat"
    assert link_text.meta["@click"] == (
        "app.open_link",
        ("https://example.test/guide?q=chat",),
    )
    linked_bold = _style(segments, "docs")
    assert linked_bold.bold
    assert not linked_bold.dim
    assert linked_bold.underline
    destination = _style(segments, "https://example.test/guide?q=chat")
    assert destination.color is not None
    assert destination.color.number == 8
    assert destination.link is None


def test_styles_fenced_code_with_bright_black_background() -> None:
    """Check closed and streaming code fences with exact content lines."""
    closed = _segments("```python\nx = **literal**\n\nprint(x)\n```")
    streaming = _segments("~~~text\nstill arriving")

    assert _plain(closed) == "x = **literal**\n\nprint(x)\n"
    assert _plain(streaming) == "still arriving\n"
    for segment in (*closed, *streaming):
        if segment.text.strip():
            assert segment.style is not None
            assert segment.style.color is None
            assert segment.style.bgcolor is not None
            assert segment.style.bgcolor.number == 8
            assert not segment.style.dim


def test_preserves_authored_blank_lines_without_block_spacing() -> None:
    """Check that block formatting does not insert or remove blank lines."""
    source = "before\n\n\n# Heading\n\nafter\n"

    assert _capture(source) == "before\n\n\n# Heading\n\nafter\n\n"


def _segments(source: str, *, width: int = 80) -> tuple[Segment, ...]:
    console = Console(width=width, color_system="standard")
    return tuple(console.render(ChatMarkdown(source)))


def _capture(source: str, *, width: int = 80) -> str:
    console = Console(width=width, color_system=None)
    with console.capture() as capture:
        console.print(ChatMarkdown(source))
    return capture.get()


def _plain(segments: tuple[Segment, ...]) -> str:
    return "".join(segment.text for segment in segments)


def _segment(segments: tuple[Segment, ...], value: str) -> Segment:
    return next(segment for segment in segments if segment.text == value)


def _segment_containing(segments: tuple[Segment, ...], value: str) -> Segment:
    return next(segment for segment in segments if value in segment.text)


def _style(segments: tuple[Segment, ...], value: str) -> Style:
    style = _segment(segments, value).style
    assert style is not None
    return style
