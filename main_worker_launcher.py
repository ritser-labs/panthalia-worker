#!/usr/bin/env python
import sys
import os
# Force Qt to use the XCB platform plugin
if sys.platform.startswith("linux"):
    os.environ["QT_QPA_PLATFORM"] = "xcb"

# When frozen, disable typeguard's runtime checking to avoid source-code lookups.
if getattr(sys, 'frozen', False):
    def dummy_typechecked(func=None, **kwargs):
        if func is None:
            return lambda f: f
        return func
    try:
        import typeguard
        typeguard.typechecked = dummy_typechecked
    except ImportError:
        # If typeguard isn't installed, nothing to patch.
        pass

import spl.worker.__main__
