import sys
sys.path.insert(1, '/home/ximo/Documents/GitHub/skforecast')

from skforecast import __version__

def test_version():
    assert __version__ ==  '0.4.1'
