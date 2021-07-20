from IPython.core.magic import (register_line_magic, register_cell_magic,
                                register_line_cell_magic)

import os
import pandas as pd
import yaml
import requests


tags_pkl = '/content/drive/Shareddrives/h2-data-science/data/tags.pkl'


@register_line_magic
def tag(line):
    '''Register each tag to this file.
    Some limitations of this are that this works line by line, so it does not collect all tags. That means we do not have a good way to delete tags from these. You can add tags anywhere.

To delete tags, you need to be able to collect the tags in a notebook, and
    delete tags from the database that are not in the current notebook. A dumb
    strategy is just delete all notebook entries for the notebook, and then
    re-add them. That would require re-running the current notebook though,
    which might not be desireable.

    Later I might add an update function that could scan the cells in each
    notebook, and just run lines with tags.


    '''

    if os.path.exists(tags_pkl):
        df = pd.read_pickle(tags_pkl)
    else:
        df = pd.DataFrame(columns=['tag', 'url'])

    # This env var will only exist while report generating
    # This is tricky
    if 'GESPYRANTO_ID' in os.environ:
        _id = os.environ['GESPYRANTO_ID']
        notebook = os.environ['GESPYRANTO_NOTEBOOK']
    else:
        # This means it is not running in the report function
        # We use the current id of the notebook.
        # copied from kora
        d = requests.get('http://172.28.0.2:9000/api/sessions').json()[0]
        _, _id = d['path'].split('=')
        notebook = d['name']

    url = f'https://colab.research.google.com/drive/{_id}'
    html = f'<a href="{url}" target="_blank">{notebook}</a>'
    for tag in line.split():
        found = (df.url == html) & (df.tag == tag)
        if not found.any():
            df.loc[len(df)] = [tag, html]

    df.to_pickle(tags_pkl)
    print(f'Registered {url} with tags: {line.split()}.')

    return line.split()


@register_cell_magic
def properties(line=None, cell=None):
    '''Cell magic to use yaml for cell properties.'''
    return yaml.load(cell, Loader=yaml.SafeLoader)


def tags(query=None):
    'This reads the tag dataframe and queries it if query is not None.'
    df = pd.read_pickle(tags_pkl)

    if query is not None:
        return df.query(query)
    else:
        return df
