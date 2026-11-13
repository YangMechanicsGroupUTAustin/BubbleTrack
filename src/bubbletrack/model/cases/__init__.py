"""Hard-case regression library: frames the app got wrong, with the right answers.

Users export a stretch of a recording as one ``.zip`` (``build``, ``bundle``);
``bubbletrack-cli eval-cases`` re-runs the current engine on every bundle
(``replay``), scores it against the user's fixes (``scoring``) and writes a
report that can be compared with an earlier one (``report``).

No Qt imports: this package belongs to the model layer.
"""

from bubbletrack.model.cases.bundle import CaseBundle, find_bundles, open_bundle, write_bundle
from bubbletrack.model.cases.schema import CaseFormatError, HardCase

__all__ = ["CaseBundle", "CaseFormatError", "HardCase", "find_bundles", "open_bundle",
           "write_bundle"]
