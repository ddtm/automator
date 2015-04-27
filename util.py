import os
import shutil
import glob

def clear_dir(path):
    try:
        shutil.rmtree(path)
    except:
        pass
    os.makedirs(path)

def get_latest(path, pattern='*'):
    found = glob.iglob(os.path.join(path, pattern))
    try:
        newest = max(found, key=os.path.getctime)
        return os.path.basename(newest)
    except:
        return None
