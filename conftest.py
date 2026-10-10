"""Repository-level pytest configuration.

Importing the vendored upstream snapshot writes ``__pycache__`` into
``causal_phase_tracking/causal_cecHT/vendor/eikeosmers_cecht/``.  That directory
is a frozen upstream snapshot and a contract test asserts it contains exactly the
approved files, so bytecode caching would make the result depend on execution
order.  Disabling bytecode writing keeps the snapshot pristine for the session.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True
