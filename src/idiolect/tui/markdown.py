"""Render focused Markdown for the local chat transcript."""

from __future__ import annotations

import re
from dataclasses import dataclass

from markdown_it import MarkdownIt
from markdown_it.tree import SyntaxTreeNode
from rich.console import Console, ConsoleOptions, RenderableType, RenderResult
from rich.padding import Padding
from rich.style import Style
from rich.text import Text


class _ChatMarkdownIt(MarkdownIt):
    """Parse only links that the terminal can open safely."""

    def validateLink(self, url: str) -> bool:
        """Accept explicit HTTP and HTTPS destinations."""
        return is_web_link(url) and super().validateLink(url)


_PARSER = _ChatMarkdownIt("zero").enable(
    (
        "fence",
        "list",
        "heading",
        "newline",
        "escape",
        "backticks",
        "emphasis",
        "link",
    )
)
_BOLD = Style(bold=True)
_ITALIC = Style(italic=True)
_CODE = Style(bgcolor="bright_black")
_METADATA = Style(color="bright_black")


def is_web_link(url: str) -> bool:
    """Return true for an explicit web link."""
    return url.casefold().startswith(("http://", "https://"))


@dataclass(frozen=True)
class _HangingText:
    """Render wrapped text below its text instead of its block marker."""

    prefix: str
    body: Text

    def __rich_console__(
        self,
        console: Console,
        options: ConsoleOptions,
    ) -> RenderResult:
        prefix_width = Text(self.prefix).cell_len
        body_width = max(options.max_width - prefix_width, 1)
        wrapped = self.body.wrap(console, body_width, overflow="fold")
        if not wrapped:
            wrapped = [Text()]
        for index, line in enumerate(wrapped):
            yield Text.assemble(
                self.prefix if index == 0 else " " * prefix_width,
                line,
            )


class ChatMarkdown:
    """Render the Markdown subset that the chat interface supports."""

    def __init__(self, source: str) -> None:
        """Parse one message without changing its source text."""
        self.source = source
        tree = SyntaxTreeNode(_PARSER.parse(source))
        lines = source.split("\n")
        self._renderables = tuple(_render_root(tree, lines))

    def __rich_console__(
        self,
        console: Console,
        options: ConsoleOptions,
    ) -> RenderResult:
        del console, options
        yield from self._renderables


def _render_root(tree: SyntaxTreeNode, source_lines: list[str]) -> list[RenderableType]:
    renderables: list[RenderableType] = []
    cursor = 0
    for node in tree.children:
        start, end = _node_map(node)
        renderables.extend(Text() for _ in range(max(start - cursor, 0)))
        renderables.extend(_render_block(node, source_lines, list_depth=0))
        cursor = end
    renderables.extend(Text() for _ in range(max(len(source_lines) - cursor, 0)))
    return renderables


def _render_block(
    node: SyntaxTreeNode,
    source_lines: list[str],
    *,
    list_depth: int,
) -> list[RenderableType]:
    if node.type == "paragraph":
        return _render_paragraph(node, source_lines)
    if node.type == "heading":
        body = _inline_child(node, _BOLD)
        prefix = f"{node.markup} " if body.plain else node.markup
        text = Text(prefix, style=_BOLD)
        text.append_text(body)
        return [text]
    if node.type == "fence":
        content_lines = node.content.split("\n")
        if node.content.endswith("\n"):
            content_lines.pop()
        return [Text(line, style=_CODE) for line in content_lines]
    if node.type in {"bullet_list", "ordered_list"}:
        return _render_list(node, source_lines, list_depth=list_depth + 1)
    return [Text("\n".join(source_lines[slice(*_node_map(node))]))]


def _render_paragraph(
    node: SyntaxTreeNode,
    source_lines: list[str],
) -> list[RenderableType]:
    renderables: list[RenderableType] = []
    start, end = _node_map(node)
    paragraph_lines = source_lines[start:end]
    rendered_lines = _inline_child(node).split("\n", allow_blank=True)
    for index, line in enumerate(rendered_lines):
        source_line = paragraph_lines[index] if index < len(paragraph_lines) else ""
        marker_count = _quote_depth(source_line)
        if marker_count == 0:
            renderables.append(line)
            continue
        content_start = _quote_content_start(line.plain, marker_count)
        prefix = f"{' ' * marker_count}{'>' * marker_count} "
        renderables.append(_HangingText(prefix, line[content_start:]))
    return renderables


def _quote_depth(source_line: str) -> int:
    match = re.match(r"^[ ]{0,3}((?:>[ \t]?)+)", source_line)
    return 0 if match is None else match.group(1).count(">")


def _quote_content_start(line: str, depth: int) -> int:
    position = 0
    for _ in range(depth):
        if line[position : position + 1] != ">":
            break
        position += 1
        if line[position : position + 1] in {" ", "\t"}:
            position += 1
    return position


def _render_list(
    node: SyntaxTreeNode,
    source_lines: list[str],
    *,
    list_depth: int,
) -> list[RenderableType]:
    renderables: list[RenderableType] = []
    for item in node.children:
        marker = item.markup if node.type == "bullet_list" else f"{item.info}{item.markup}"
        prefix = f"{' ' * list_depth}{marker} "
        children = item.children
        cursor = _node_map(item)[0]
        first_is_paragraph = bool(children and children[0].type == "paragraph")
        if first_is_paragraph:
            first = children[0]
            renderables.append(_HangingText(prefix, _inline_child(first)))
            cursor = _node_map(first)[1]
            children = children[1:]
        else:
            renderables.append(Text(prefix.rstrip()))
            cursor += 1

        continuation_indent = Text(prefix).cell_len
        for child in children:
            start, end = _node_map(child)
            renderables.extend(Text() for _ in range(max(start - cursor, 0)))
            if child.type in {"bullet_list", "ordered_list"}:
                renderables.extend(
                    _render_list(child, source_lines, list_depth=list_depth + 1)
                )
            else:
                renderables.extend(
                    Padding(renderable, (0, 0, 0, continuation_indent))
                    for renderable in _render_block(
                        child,
                        source_lines,
                        list_depth=list_depth,
                    )
                )
            cursor = end

        item_end = _node_map(item)[1]
        renderables.extend(Text() for _ in range(max(item_end - cursor, 0)))
    return renderables


def _inline_child(node: SyntaxTreeNode, style: Style | None = None) -> Text:
    inline = next((child for child in node.children if child.type == "inline"), None)
    return Text() if inline is None else _inline_text(inline, style or Style())


def _inline_text(node: SyntaxTreeNode, style: Style) -> Text:
    text = Text()
    for child in node.children:
        if child.type == "text":
            text.append(child.content, style)
        elif child.type in {"softbreak", "hardbreak"}:
            text.append("\n")
        elif child.type == "code_inline":
            text.append(child.content, style + _CODE)
        elif child.type == "strong":
            text.append_text(_inline_text(child, style + _BOLD))
        elif child.type == "em":
            text.append_text(_inline_text(child, style + _ITALIC))
        elif child.type == "link":
            href = str(child.attrGet("href") or "")
            text.append_text(
                _inline_text(
                    child,
                    style
                    + Style(
                        underline=True,
                        link=href,
                        meta={"@click": ("app.open_link", (href,))},
                    ),
                )
            )
            text.append(" (", style + _METADATA)
            text.append(href, style + _METADATA)
            text.append(")", style + _METADATA)
        else:
            text.append(child.content, style)
    return text


def _node_map(node: SyntaxTreeNode) -> tuple[int, int]:
    mapping = node.map
    if mapping is None:
        return (0, 0)
    return mapping
