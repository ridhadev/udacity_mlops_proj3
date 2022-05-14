import sys
import os
pythonpath = os.path.realpath(
        os.path.join(
            os.path.dirname(__file__),
            '..',
            'starter')
    )
print(f"Python path {pythonpath}")
sys.path.append(pythonpath)