from IPython.display import HTML
import subprocess as sp
import os


def get_url(path):
    '''Return an HTML URL to the PATH. This may not be fully functional, GDrive uses
    a lot of different paths and for different purposes, e.g. to view, download,
    etc.

    '''

    _id = sp.getoutput(f"xattr -p 'user.drive.id' \"{path}\"")

    if os.path.isdir(path):
        url = f'https://drive.google.com/drive/u/0/folders/{_id}'
    elif path.endswith('ipynb'):
        url = f'https://colab.research.google.com/drive/{_id}'
    else:
        url = f'https://drive.google.com/file/d/{_id}'

    return HTML(f'<a href="{url}" target="_blank">{path}</a>')
