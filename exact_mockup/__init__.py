"""Put real pages and covers into AI-generated mockups, with exact text."""
from .cover import paste_cover
from .pages import paste_pages

__all__ = ["paste_pages", "paste_cover"]
__version__ = "0.1.0"
