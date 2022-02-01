'''Plugin class to read image and mmolH data.
Original version: https://github.com/espyranto/espyranto/blob/master/espyranto/g3/umolH.py

This is new version for colab.
'''
import gespyranto

from datetime import datetime
import glob
import operator

import os
import numpy as np

import json
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy.interpolate import UnivariateSpline

import subprocess as sp

import shutil
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


class umolH:
    def __init__(self, plate):
        self.name = 'umolH'  # you need this so you know what data you have.
        self.plate = plate
        # this is umolH vs time in each well. Units are micromoles H2
        datafile = os.path.join(plate.path,
                                'output/wholeplatedat.JSON')
        if not os.path.exists(datafile):
            raise Exception(f'{datafile} does not exist')

        with open(datafile) as f:
            self.umolH_ef = json.loads(f.read())

        self.umolH = np.array([self.umolH_ef[i]['umol H2']
                               for i in range(len(self.umolH_ef))])

        self.umolH_max_rate = np.array([self.umolH_ef[i]['Max Rate (umol/h)']
                                        for i in range(len(self.umolH_ef))])
        self.umolH_max_rate = (self.umolH_max_rate.reshape(self.plate.nrows,
                                                           self.plate.ncols)
                               .flatten())

        self.max = np.array([self.umolH_ef[i]['Max H2 (umol)']
                             for i in range(len(self.umolH_ef))])
        self.max = self.max.reshape(self.plate.nrows,
                                    self.plate.ncols).flatten()

        # Array of images and timestamps
        image_files = glob.glob(os.path.join(plate.path, 'images/*.jpg'))
        if not len(image_files) > 0:
            raise Exception('No image file found')

        # These are the timestamps for which the images were taken but they are
        # not the same as reaction times because the reactor is not
        # illuminated by blue light while taking the image.
        dates = [datetime.strptime(os.path.split(f)[-1].split('_')[-1],
                           f'y%ym%md%dH%HM%MS%S.jpg')
            for f in image_files]

        # This is a sorted list (by date) of tuples(filename, datetime) for
        # each image
        self.images = sorted(zip(image_files, dates),
                             key=operator.itemgetter(1))

        # Sometimes there are under images.
        under_files = glob.glob(os.path.join(plate.path, 'under/*.jpg'))
        if len(under_files) == 0:
            self.under_images = []
        else:
            # It looks like there is no A in these, just _
            dates = [datetime.strptime(os.path.split(f)[-1].split('_')[-1],
                                       f'y%ym%md%dH%HM%MS%S.jpg')
                     for f in under_files]

            # This is a sorted list (by date) of tuples(filename, datetime) for
            # each image
            self.under_images = sorted(zip(under_files, dates),
                                       key=operator.itemgetter(1))

        # And sometimes sidebyside
        # Sometimes there are under images.
        side_files = glob.glob(os.path.join(plate.path, 'sidebyside/*.jpg'))
        if len(side_files) == 0:
            self.side_images = []
        else:
            # It looks like there is no A in these, just _
            dates = [datetime.strptime(os.path.split(f)[-1].split('_')[-1],
                                       f'y%ym%md%dH%HM%MS%S.jpg')
                     for f in side_files]

            # This is a sorted list (by date) of tuples(filename, datetime) for
            # each image
            self.side_images = sorted(zip(side_files, dates),
                                      key=operator.itemgetter(1))

        self.timestep = self.plate.metadata['Picture time (min)'] * 60

    def __repr__(self):
        '''This is a string to summarize this data.'''
        s = ['', 'umolH data']
        s += [f' {len(self.images)} images were acquired.',
              f' Start time: {self.images[0][1]}',
              f' End time: {self.images[-1][1]}',
              f' The timestep is {self.timestep} sec']
        s += [f'umolH data has shape: {self.umolH.shape}']
        return '\n'.join(s)

    def _repr_html_(self):
        # get id with xattr.
        xattr = shutil.which('xattr')
        if xattr is None:
            return repr(self)

        _id = sp.getoutput(f"xattr -p 'user.drive.id' \"{self.plate.path}\"")
        url = f'https://drive.google.com/drive/u/0/folders/{_id}'
        return f'''
      <a href="{url}" target="_blank">{self.plate.path}</a><br>
        umolH data<br>
        {len(self.images)} images were acquired. <br>
        Start time: {self.images[0][1]}<br>
        End time: {self.images[-1][1]}<br>
        The timestep is {self.timestep} sec<br>
        umolH data has shape: {self.umolH.shape}'''

    def plot_umolH_grid(self):
        '''Make a grid-plot of umolH data.'''
        fig = make_subplots(rows=self.plate.nrows, cols=self.plate.ncols,
                            shared_xaxes=True, shared_yaxes=True,
                            vertical_spacing=0.02,
                            x_title='Time (h)',
                            y_title='H<sub>2</sub> Produced (umol)')
        t = np.arange(0, self.umolH.shape[1]) * self.timestep / 3600  # hours

        for row in range(self.plate.nrows):
            for col in range(self.plate.ncols):
                ind = row*self.plate.ncols+col
                # smooth data to get the rate. s is a tunable parameter that
                # specifies how much smoothing I don't have a great way to
                # choose it. s=8 worked for an example.
                y = self.umolH[ind]
                spl = UnivariateSpline(t, y, s=8)
                rate = spl.derivative()(t)
                maxrate_ind = rate.argmax()

                # smoothed data
                fig.add_trace(go.Scatter(x=t, y=spl(t),
                                         line=dict(color="#0000ff", width=1)),
                              row=row + 1, col=col + 1)
                # raw data comes from concentration pivot table
                # We join this list with breaks in the trace.
                # Make concentrations greater than zero be black
                # so they stand out better.
                hovertext = []
                # for s, v in self.plate.pc.loc[ind].items():
                #     if v > 0:
                #         hovertext += [f'<font color="black">{s[1]}: {v:4.2}</font>']
                #     else:
                #         hovertext += [f'{s[1]}: {v:4.2f}']
                # The colored text did not work, so now I only show solutions greater than 0
                for s, v in self.plate.pc.loc[ind].items():
                    if v > 0:
                        hovertext += [f'{s[1]}: {v:4.2f}']
                hovertext += [f'Attributes: {self.plate.uniqueatt[ind]}']
                hovertext += [f'umolH Max: {self.max[ind]:4.2f}']

                fig.add_trace(go.Scatter(x=t, y=y,
                                         text='<br>'.join(hovertext),
                                         hoverinfo='text',
                                         marker=dict(symbol='circle',
                                                     size=1),
                                         line=dict(color="#0000ff")),
                              row=row + 1, col=col + 1)
                # Plot maxH as red circle
                max_ind = self.umolH[ind].argmax()
                fig.add_trace(go.Scatter(x=[t[max_ind]],
                                         y=[self.umolH[ind][max_ind]],
                                         line=dict(color='red')),
                              row=row + 1, col=col + 1)
                # Plot max rate from spline in green
                fig.add_trace(go.Scatter(x=[t[maxrate_ind]],
                                         y=[self.umolH[ind][maxrate_ind]],
                                         marker=dict(symbol='square'),
                                         line=dict(color='green')),
                              row=row + 1, col=col + 1)

        fig.update_layout(height=900, width=800, font=dict(size=8))
        fig.update_yaxes(range=[np.min(self.umolH), np.max(self.umolH)])

        _id = sp.getoutput(f"xattr -p 'user.drive.id' \"{self.plate.path}\"")
        url = f'https://drive.google.com/drive/u/0/folders/{_id}'
        fig.update_layout(title=f'<a href="{url}" '
                          f'target="_blank">{self.plate.path}</a>')
        fig.update_layout(showlegend=False)
        return fig.show()

    def plot_hmax_heatmap(self):
        '''Plot a heatmap of the maximum hydrogen produced in each well.'''
        tooltips = []
        nrows, ncols = self.plate.nrows, self.plate.ncols
        for row in range(nrows):
            S = []
            for col in range(ncols):
                s = ''
                ind = ncols * row + col
                d = self.plate.pc.iloc[ind].to_dict()
                for key in d:
                    s += f'<br>{key[1]}: {d[key]:1.2f}'
                s += f'<br>Attributes: {self.plate.uniqueatt[ind]}'
                s += f'<br>umolH Max: {self.max[ind]:1.2f}'
                S += [s]
            tooltips += [S]
        maxH = self.max.reshape(nrows, ncols)
        fig = go.Figure(go.Heatmap(z=maxH,
                                   zmin=0, zmax=25,
                                   hovertemplate='%{text}<extra></extra>',
                                   text=tooltips,
                                   hoverinfo='text'))
        fig.update_xaxes(title='columns')
        fig.update_yaxes(autorange="reversed", title='rows')
        fig.update_layout(title={'text': 'Max H<sub>2</sub> (umol)'})
        return fig.show()

    def plot_hmaxrate_heatmap(self):
        '''Plot a heatmap of the maximum rate in each well.'''
        tooltips = []
        nrows, ncols = self.plate.nrows, self.plate.ncols
        for row in range(nrows):
            S = []
            for col in range(ncols):
                s = ''
                ind = ncols * row + col
                d = self.plate.pc.iloc[ind].to_dict()
                for key in d:
                    s += f'<br>{key[1]}: {d[key]:1.2f}'
                s += f'<br>Attributes: {self.plate.uniqueatt[ind]}'
                s += f'<br>umolH Max: {self.max[ind]:1.2f}'
                S += [s]
            tooltips += [S]
        maxrateH = self.umolH_max_rate.reshape(nrows, ncols)
        fig = go.Figure(go.Heatmap(z=maxrateH,
                                   zmin=0, zmax=25,
                                   hovertemplate='%{text}<extra></extra>',
                                   text=tooltips,
                                   hoverinfo='text'))
        fig.update_xaxes(title='columns')
        fig.update_yaxes(autorange="reversed", title='rows')
        fig.update_layout(title={'text': 'Max Rate H<sub>2</sub> (umol/hr)'})
        return fig.show()

    def first_last_plate_images(self):
        '''Makes a subplot of first and last plates for the report.'''
        img0 = mpimg.imread(self.images[0][0])
        img1 = mpimg.imread(self.images[-1][0])
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(img0)
        ax[0].set(title='t=0', xticklabels=[], yticklabels=[])
        ax[1].imshow(img1)
        tf = self.umolH.shape[1] * self.timestep / 3600
        ax[1].set(title=f't={tf:1.1f} hr', xticklabels=[], yticklabels=[])
        return fig, ax

    def first_last_plate_under_images(self):
        '''Makes a subplot of first and last under plates for the report.'''
        if len(self.under_images) == 0:
            print('No under images available.')
            return None

        img0 = mpimg.imread(self.under_images[0][0])
        img1 = mpimg.imread(self.under_images[-1][0])
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(img0)
        ax[0].set(title='t=0', xticklabels=[], yticklabels=[])
        ax[1].imshow(img1)
        tf = self.umolH.shape[1] * self.timestep / 3600
        ax[1].set(title=f't={tf:1.1f} hr', xticklabels=[], yticklabels=[])
        return fig, ax

    def first_last_plate_sidebyside_images(self):
        '''Makes a subplot of first and last sidebyside plates for the report.'''
        if len(self.side_images) == 0:
            print('No under images available.')
            return None

        img0 = mpimg.imread(self.side_images[0][0])
        img1 = mpimg.imread(self.side_images[-1][0])
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(img0)
        ax[0].set(title='t=0', xticklabels=[], yticklabels=[])
        ax[1].imshow(img1)
        tf = self.umolH.shape[1] * self.timestep / 3600
        ax[1].set(title=f't={tf:1.1f} hr', xticklabels=[], yticklabels=[])
        return fig, ax

    def show_plate(self, i):
        '''Show an image of the plate at the ith timestep. I is normally an integer, but
        it can also be a slice. p.show_plate(slice(0, 3)) to show the first
        three images. This is kind of a placeholder. You probably want to add
        some control for the size of images with slices.

        '''
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        if isinstance(i, int):
            img = mpimg.imread(self.images[i][0])
            return plt.imshow(img)
        elif isinstance(i, slice):
            inds = list(range(*i.indices(len(self.images))))
            N = len(inds)
            f, axes = plt.subplots(N)
            for j in range(N):
                axes[j].imshow(mpimg.imread(self.images[inds[j]][0]))
            return f, axes

    def dataframe(self):
        t = np.arange(self.umolH.shape[1]) * self.timestep / 3600
        df = self.plate.pc.copy()
        df.columns = df.columns.droplevel()
        df['umolH_max_rate'] = self.umolH_max_rate
        df['umolH_max'] = self.max
        df['umolh'] = [np.array(x) for x in self.umolH.tolist()]
        df['time'] = [np.array(x) for x in np.repeat(t[None,:],len(df), axis = 0)]
        df['attributes'] = self.plate.uniqueatt
        df['directory'] = self.plate.path
        return df


gespyranto.plate.Plate.datamodules += [umolH]
print('Registered umolH')
