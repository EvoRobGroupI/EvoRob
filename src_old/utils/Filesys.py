import os
from pathlib import Path
import re


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent


def search_file_list(rootname, file_name):
    file_list = []
    for root, dirs, files in os.walk(rootname):
        for file in files:
            if file_name in file:
                file_list.append(os.path.join(root, file))
    file_list.sort(key=natural_keys)
    return file_list


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]