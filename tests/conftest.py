import sys
import os

""" Add the starter folder to python path for pytest runs"""
pythonpath = os.path.realpath(
        os.path.join(
            os.path.dirname(__file__),
            '..',
            'starter')
    )

sys.path.append(pythonpath)
