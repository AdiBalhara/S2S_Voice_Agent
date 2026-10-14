"""TrijoTech Voice Sales Agent Application."""

from app.database.connection import configure_windows_event_loop

# Ensure compatible event loop on Windows
configure_windows_event_loop()
