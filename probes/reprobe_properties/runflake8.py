import os
import sys

os.chdir(sys.argv[1])
from flake8.main.cli import main
raise SystemExit(main(sys.argv[2:]))
