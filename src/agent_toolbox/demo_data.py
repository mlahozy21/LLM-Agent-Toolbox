"""A small tool registry and labelled queries used by the demo and the eval."""

from .tool_selection import Tool

TOOLS = [
    Tool("web_search", "Search the web for current information and news."),
    Tool("calculator", "Evaluate arithmetic and mathematical expressions."),
    Tool("weather", "Get the current weather and forecast for a city."),
    Tool("calendar", "Create, read and manage calendar events and reminders."),
    Tool("email_send", "Compose and send an email to one or more recipients."),
    Tool("translate", "Translate text between languages."),
    Tool("sql_query", "Run a SQL query against the analytics database."),
    Tool("code_exec", "Execute a Python code snippet and return its output."),
    Tool("image_gen", "Generate an image from a text prompt."),
    Tool("file_read", "Read the contents of a file from disk."),
    Tool("stock_price", "Look up the current price of a stock ticker."),
    Tool("unit_convert", "Convert between physical units (length, mass, time)."),
]

# (query, set-of-relevant-tool-names) — ground truth for recall@k
LABELLED = [
    ("What's the weather in Paris tomorrow?", {"weather"}),
    ("How much is 23.5 times 88?", {"calculator"}),
    ("Translate 'good morning' into Japanese", {"translate"}),
    ("Send a message to my boss about the meeting", {"email_send"}),
    ("What is Apple's share price right now?", {"stock_price"}),
    ("Schedule a dentist appointment next Monday", {"calendar"}),
    ("Run this query to count users by country", {"sql_query"}),
    ("Find the latest news about the elections", {"web_search"}),
    ("Convert 10 kilometres to miles", {"unit_convert"}),
    ("Draw a picture of a red bicycle", {"image_gen"}),
]
