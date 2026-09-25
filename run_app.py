"""AutoCut Standalone Application Launcher."""

import sys
from autocut.cli import main

if __name__ == "__main__":
    is_direct_launch = len(sys.argv) <= 1
    if is_direct_launch:
        sys.argv.append("record")

    try:
        main()
    except SystemExit as e:
        if is_direct_launch and e.code != 0:
            try:
                input("\n[Process exited with code %s. Press ENTER to close window...]" % e.code)
            except Exception:
                pass
        raise
    except Exception as exc:
        print(f"\n[FATAL ERROR] {exc}")
        if is_direct_launch:
            try:
                input("\n[Press ENTER to close window...]")
            except Exception:
                pass
        sys.exit(1)
