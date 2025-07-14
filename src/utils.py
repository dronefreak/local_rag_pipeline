from rich.console import Console
from rich.theme import Theme


def get_rich_console() -> Console:
    """Pretty printing console provided the python lib called "rich".

    Returns:
        Console: rich.console.Console type printer function that can
                 be used instead of regular print.
    """
    # Set rich themes for console
    custom_theme = Theme(
        {
            "info": "bold bright_green",
            "warning": "bold bright_yellow",
            "danger": "bold bright_red",
            "summary": "italic green",
        }
    )
    console = Console(theme=custom_theme)
    return console


def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)
