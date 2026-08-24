import os, sys
# this launcher lives in uis/; add the repo root so project imports below resolve
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import configs.config_general as cnf
from uis.ui_labeling import startUi

if __name__ == '__main__':
    print(cnf.BASE_DIR)
    startUi()
    
