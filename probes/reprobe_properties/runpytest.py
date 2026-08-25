import os
import sys

os.chdir(sys.argv[1])
import pytest
raise SystemExit(pytest.main(sys.argv[2:]))
