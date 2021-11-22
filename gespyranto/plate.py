'''
The original verison: https://github.com/espyranto/espyranto/blob/master/espyranto/g3/plate.py

This is a new version for colab.
'''
import glob
import os
import numpy as np
import pandas as pd
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor
import gespyranto
from gespyranto.colab import get_url
import yaml
from IPython.display import HTML
import subprocess as sp
import time
import pickle
import sys
from tabulate import tabulate


class Plate:

    datamodules = []  # this will be added to by plugins eg. umolH

    def __init__(self, path=None, ncols=12):
        '''Read the data in the directory at PATH.
        NCOLS is number of columns in the plate.'''
        self.path = path
        self.ncols = ncols

        if path.endswith('/'):
            path = path[:-1]
            
        #See if there are outputs or just inputs
        #output = os.path.join(path, 'output/')
        #if not os.path.exists(output):
        #    sys.exit()
        
        params = os.path.join(path, 'input/Parameters.xlsx')
        if not os.path.exists(params):
            raise Exception(f'{params} does not exist')
        
        output = os.path.join(path, 'output/')
        try: 
            os.path.exists(output)
        except Exception:
            pass 

        # extract metadata and solution components from the Parameters file
        p = self.parameters = pd.read_excel(params, engine='openpyxl')
        solution_labels = p['Solutions'].dropna().values.tolist()
        concentrations = p['Stock Conc (mM)'].dropna().values.tolist()
        param_list = [x.strip(' ') for x in self.parameters['Parameters'].dropna().values.tolist()]
        values_list = [x.strip(' ') for x in self.parameters['Unnamed: 1'].dropna().values.tolist()]
        pars = dict(zip(param_list, values_list))
        self.metadata = dict(**pars,
                             Solutions=solution_labels,
                             Concentrations=concentrations)

                # Number of wells is based on the reactor
        if self.metadata['Reactor'] in ['G', 'R']:
            self.nwells = 108
        if self.metadata['Reactor'] in ['V']:
            self.nwells = 96
        self.nrows = int(self.nwells/self.ncols)

        # Get array of strings containing unique well attribute information
        # Previous experiments not containing plate_design files will output in situ here
        layout = os.path.join(path, 'input/plate_design.xlsx')
        if os.path.exists(layout):
            self.uniqueatt = pd.read_excel(layout, header=None).loc[0:self.nrows-1].to_numpy().flatten()
            description = pd.read_excel(layout, skiprows=range(0, int(self.nrows)),
                                        header=None)[0].values[0] #takes the first and only element in df
            self.metadata['Description'] = description
            # creates dataframe for plate layout
            self.plate_layout = pd.read_excel(layout, header=None).loc[0:self.nrows-1]
            self.plate_layout.rename_axis('Rows', axis=0, inplace=True)  # label index as rows
            self.plate_layout.rename_axis('Columns', axis=1, inplace=True) # label the column names as columns

        if not os.path.exists(layout):
            self.uniqueatt = np.repeat(['In Situ'], self.nwells) 
            self.metadata['Description'] = 'Screening experiment'
            self.plate_layout = pd.DataFrame(data = self.uniqueatt.reshape(self.nrows, self.ncols))

        # Now we build a long DataFrame from the Robot files
        df = pd.DataFrame()

        for i, label in enumerate(self.metadata['Solutions']):
            # these files are named like 20_2_label.xls the _label means it
            # it is the label solution
            if os.path.exists(f'{path}/input/TA_{label}.xls'):
                xlspattern = f'{path}/input/TA_{label}.xls'
            elif os.path.exists(f'{path}/input/TB_{label}.xls'):
                xlspattern = f'{path}/input/TB_{label}.xls'
            else:
                xlspattern = f'{path}/input/*[0-9]_{label}.xls'
            xls = glob.glob(xlspattern)  # there should be one match
            if len(xls) == 1:
                xls = xls[0]
            else:
                raise Exception('Wrong number of xls files found for'
                                f' {xlspattern}',
                                f'{xls}')

            # This goes down the columns. So the second element is row 1,
            # column 0. I want the order to go across rows instead
            # arr.reshape(12, 9).T.flatten()
            vols = pd.read_excel(xls)['Volume'].values
            vols = vols.reshape(ncols, -1).T.flatten()
            
            #If an internal standard exists, must add a ninth row to vols
            if 'Internal Standard' in self.metadata:
                vols = np.concatenate([vols, np.zeros(self.nwells-len(vols))])
                
            wd = pd.DataFrame()
            wd['Well-number'] = np.arange(0, len(vols))
            wd['Solution'] = label
            wd['Volume'] = vols

            df = pd.concat([df, pd.DataFrame(wd)])

        df.columns = ['Well-number', 'Solution', 'Volume']
        self.nrows = len(vols) // ncols

        self.df = df

        # Pivot table to get volume of each solution
        self.pv = self.df.pivot_table(index=['Well-number'],
                                      columns='Solution')

        # Reorder columns so they are in the same order as the solutions.
        self.pv = self.pv.reindex([('Volume', label)
                                   for label in self.metadata['Solutions']],
                                  axis=1)

        # Pivot table on concentrations
        total_volume = self.pv.sum(axis=1).iloc[0]
        self.pc = (self.pv * self.metadata['Concentrations']) / total_volume
        self.pc = self.pc.rename(columns={'Volume': 'Concentration'})

        # Last we load the data
        self.data = {}
        for module in self.datamodules:
            mod = module(self)  # This is an instance
            self.data[mod.name] = mod


    @property
    def url(self):
        _id = sp.getoutput(f"xattr -p 'user.drive.id' \"{self.path}\"")
        url = f'https://drive.google.com/drive/u/0/folders/{_id}'
        return HTML(f'<a href="{url}" target="_blank" title="{self!r}">{self.path}</a>')

    def __repr__(self):
        '''Representation'''
        s = [f'{self.path}']
        for key in self.metadata:
            s += [str(key), str(self.metadata[key]), '']
        for key in self.data:
            s += [str(self.data[key]), '']
        return '\n'.join(s)

    def _repr_html_(self):
        url = get_url(self.path)
        s = f'''<a href="{url}" target="_blank">{self.path}</a>'''
        s += f'<br>Report {self.report().data}<br>'
        for key in self.metadata:
            s += f'<br>{key}: {self.metadata[key]}'
        for key in self.data:
            s += f'<br><pre>{self.data[key]}</pre>'
        s += f'<br> Plate Layout <br>'
        #s+= tabulate(self.plate_layout, tablefmt="html", headers="keys")
        s += self.plate_layout.to_html()
        return s

    def report(self, update=False, verbose=False):
        '''Generate the report.
        If update is True, regenerate the report.
        Returns url to the report.'''

        report_file = os.path.join(self.path, f'readme-{os.path.split(self.path)[1]}.ipynb')

        # Make a temp version so we can get an id. This only happens when there
        # is not a report
        if not os.path.exists(report_file):
            nb = nbf.v4.new_notebook()
            nb['cells'] = []
            with open(report_file, 'wt') as f:
                nbf.write(nb, f)

        # For new files, it seems to take some time to get the drive id. I give
        # it 15 seconds here. It seems to take about 9-10 seconds after it is
        # made for the first time. It is a lot faster after that.
        n = 0
        _id = 'local'
        while 'local' in _id and n < 15:
            _id = sp.getoutput(f"xattr -p 'user.drive.id' \"{report_file}\"")

            if 'local' not in _id:
                break
            if verbose:
                print(f'{n}. local in {_id}, trying again')
            n += 1
            time.sleep(1)

        if 'local' in _id:
            raise Exception('no id found')

        # This is for setting tags to the right value
        os.environ['GESPYRANTO_ID'] = _id
        os.environ['GESPYRANTO_NOTEBOOK'] = os.path.split(report_file)[1]
        if os.path.exists(report_file) and not update:
            pass
        else:
            # This is pretty ugly, a template loaded externally might be easier,
            # although I am not sure how to get all the cells set up there,
            # unless it is a json file to start with.
            nb = nbf.v4.new_notebook()
            tags = ' '.join(self.metadata['Solutions'])
            properties = yaml.dump(self.metadata)
            nb['cells'] = [nbf.v4.new_markdown_cell(f'''# {os.path.split(self.path)[1]}
This report was auto-generated by gespyranto on {time.asctime()}. Any edits will be lost on the next generation.
'''),

                           nbf.v4.new_code_cell('''exec(open("/content/drive/Shareddrives/h2-data-science/users/jkitchin/python/gdrive-setup.py").read())
# Load gespyranto
from gespyranto.plate import Plate
from gespyranto.umolh import umolH'''),
                           nbf.v4.new_markdown_cell('# Metadata'),
                           nbf.v4.new_code_cell(f'%tag {tags}'),
                           nbf.v4.new_code_cell(f'''%%properties
{properties}'''),

                           nbf.v4.new_markdown_cell('# Plate'),

                           nbf.v4.new_code_cell(f'''p = Plate('{self.path}')
p'''),

                           nbf.v4.new_markdown_cell('''## H<sub>2</sub> production vs time
<font color="green">Green square is max rate</font>, <font color="red">Red circle is max H produced</font>'''),
                           nbf.v4.new_code_cell("p.data['umolH'].plot_umolH_grid()"),
                           nbf.v4.new_markdown_cell('## Heatmaps'),
                           nbf.v4.new_code_cell("p.data['umolH'].plot_hmax_heatmap()"),
                           nbf.v4.new_code_cell("p.data['umolH'].plot_hmaxrate_heatmap()"),
                           nbf.v4.new_markdown_cell('## Top plate images'),
                           nbf.v4.new_code_cell("p.data['umolH'].first_last_plate_images();"),
                           nbf.v4.new_markdown_cell('## Under plate images'),
                           nbf.v4.new_code_cell("p.data['umolH'].first_last_plate_under_images();"),
                           nbf.v4.new_markdown_cell('## Sidebyside plate images'),
                           nbf.v4.new_code_cell("p.data['umolH'].first_last_plate_sidebyside_images();"),]

            ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
            ep.preprocess(nb, {})

            del os.environ['GESPYRANTO_ID']
            del os.environ['GESPYRANTO_NOTEBOOK']

            with open(report_file, 'wt') as f:
                nbf.write(nb, f)
            if verbose:
                print(f'Wrote {report_file}')

        url = f'https://colab.research.google.com/drive/{_id}'
        return HTML(f'<a href="{url}" target="_blank" title="{self!r}">{report_file}</a>')


    # def __getitem__(self,index):
    #     '''get self[index]

    #     if index is an integer, it is a linear index that is converted to a row,column
    #     if index is (row, col) return that well data.
    #     The data classes are responsible for implementing indexing this way.
    #     Slicing is not currently supported.'''

    #     if isinstance(index,int):
    #         row = index // self.ncols
    #         col = index % self.ncols

    #     elif isinstance(index, tuple) and len(index) == 2:
    #         row, col = index
    #         index = row * self.ncols + col
    #     else:
    #         raise Exception('index is the wrong shape. It should be like p[0] or p[0, 0].')

    #     return {'row': row,
    #             'col': col,
    #             'metals': self.contents[index],
    #             'data': {d.name: d[index] for d in self.data}}






class Plates:
    data_root = '/content/drive/Shareddrives/h2-data-science/data'
    plates_pkl = '/content/drive/Shareddrives/h2-data-science/data/plates.pkl'

    def __init__(self, update=False):
        if not os.path.exists(self.plates_pkl) or update:
            self.plates = self.find_plate_directories()
        else:
            self.plates = self.load_plates(update=update)

    def __repr__(self):
        tags = []
        for p in self.plates.values():
            tags += p.metadata['Solutions']

        tags = set(tags)
        return f'{len(self.plates)} plates with tags: {tags}.'

    def find_plate_directories(self):
        'Walk the data root directory and return a dictionary of plates.'

        plate_directories = []
        for root, dirs, files in os.walk(self.data_root):
            for dir in dirs:
                if (os.path.isdir(os.path.join(root, dir, 'input'))
                    and os.path.isdir(os.path.join(root, dir, 'output'))
                    and os.path.isdir(os.path.join(root, dir, 'images'))):
                    plate_directories += [(os.path.join(root, dir))]
        print(f'Found {len(plate_directories)} directories')
        plates = {pd: Plate(pd) for pd in plate_directories}
        pickle.dump(plates, open(self.plates_pkl, 'wb'))
        return plates

    def load_plates(self, update=False):
        'Load the toc dataframe.'
        if update:
            plates = self.find_plate_directories()
        else:
            plates = pickle.load(open(self.plates_pkl, 'rb'))
        print(f'Loaded {len(plates)} plates.')
        return plates

    def toc(self, update=False):
        'Generate the table of contents for the plates'
        for _, plate in self.plates.items():
            display(plate.report(update=update))

    def index(self, remove=()):
        'Display an index.'
        tags = []
        for p in self.plates.values():
            tags += p.metadata['Solutions']

        tags = set(tags)

        index = {}
        for p in self.plates.values():
            for tag in p.metadata['Solutions']:
                if tag in index:
                    index[tag] += [p.url]
                else:
                    index[tag] = [p.url]

        keys = list(index.keys())
        for k in remove:
            keys.remove(k)

        keys.sort(key=str.lower)
        for key in keys:
            print(key)
            for plate in index[key]:
                display(plate)
        print()
        return index

    def tags(self):
        "Return the tag dataframe"
        return pd.read_pickle(gespyranto.tags_pkl)
