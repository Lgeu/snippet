from __future__ import division
import sys
PYTHON3 = sys.version_info.major == 3
if not PYTHON3:
    from itertools import izip as zip
    range = xrange
    input = raw_input
