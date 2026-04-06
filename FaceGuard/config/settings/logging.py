# Đây là tệp python định nghĩa cấu hình logging cho FaceGuard. Nó thiết lập logging để ghi lại thông tin về quá trình chạy của ứng dụng, bao gồm cả lỗi và thông tin debug. Hiện tại tệp này đang trống, cần viết thêm mã dựa trên các tệp đang tồn tại trong dự án 
"""
Logging Configuration Constants.
These labels are designed to trigger VSColorOutput64 patterns in Visual Studio.
"""

from __future__ import annotations

from typing import Iterable, List

try:
    from rich.console import Console
    from rich.table import Table
except ImportError:
    Console = None
    Table = None

# ==================== LOG LEVEL LABELS ====================

# Nhóm: LogError (Regex 14) -> RED
ERROR       = "[ERROR]"
CRITICAL    = "[CRITICAL]"
FAIL        = "[FAIL]"

# Nhóm: LogWarning (Regex 15) -> YELLOW
WARNING     = "[WARNING]"
SKIP        = "[SKIP]"

# Nhóm: LogCustom1 (Regex 16) -> GREEN
SUCCESS     = "[SUCCESS]"
PASS        = "[PASS]"
SAVE        = "[SAVE]"
UPLOAD      = "[UPLOAD]"

# Nhóm: LogCustom2 (Regex 17) -> BLUE/CYAN
PHASE       = "[PHASE]"
STAGE       = "[STAGE]"
STEP        = "[STEP]"

# Nhóm: LogInformation (Regex 18) -> CYAN
INFO        = "[INFO]"
METADATA    = "[METADATA]"
DEBUG       = "[DEBUG]"
TEST        = "[TEST]"
LOAD        = "[LOAD]"
DOWNLOAD    = "[DOWNLOAD]"


# ==================== CONSOLE HELPERS ====================

if Console is not None:
    console = Console()
else:
    class _FallbackConsole:
        @staticmethod
        def print(*args, **kwargs):
            print(*args)

    console = _FallbackConsole()


def create_table(title: str, columns: List[str], rows: Iterable[Iterable[str]]):
    """Create a display table for CLI/terminal output."""
    if Table is None:
        lines = [f"== {title} ==", " | ".join(columns)]
        for row in rows:
            lines.append(" | ".join(str(c) for c in row))
        return "\n".join(lines)

    table = Table(title=title)
    for col in columns:
        table.add_column(str(col))
    for row in rows:
        table.add_row(*[str(c) for c in row])
    return table


def print_section_header(title: str) -> None:
    """Print a standard section header."""
    console.print("\n" + "=" * 70)
    console.print(f"  {title}")
    console.print("=" * 70)

# ==================== EXPORT ALL ====================
__all__ = [
    "ERROR", "CRITICAL", "FAIL",
    "WARNING", "SKIP",
    "SUCCESS", "PASS", "SAVE", "UPLOAD",
    "PHASE", "STAGE", "STEP",
    "INFO", "METADATA", "DEBUG", "TEST", "LOAD", "DOWNLOAD",
    "console", "create_table", "print_section_header",
]