import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from scipy.optimize import minimize 
from statsmodels.formula.api import ols
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from pyDOE2 import *
from ase.data import chemical_symbols

class analysis:
      '''
    Function takes in a dataframe, metals, components, and a target variable.
    Outputs model and ANOVA summary. Creates polynomial features with sklearn store input as global variables
    
    dir is a list of directory paths
    metals is a list of metals of interest 
    components are the potential important solutions in the experiment and dataframe
    target is the target variable 
      '''
      def __init__(self, df, metals, components, target):
        '''store input as global variables
        dataframe is the complete dataframe generated by gespyranto for the experiments that we are looking at
        metals is a list of metals of interest 
        components are the potential important solutions in the experiment and dataframe
        target is the target variable '''

        self.metals = metals
        self.comp = components
        self.target = target
    
        self.df = df
        self.df = self.df.fillna(0)
        
        #create dataframe that focuses on the self.metals
        ce = '|'.join(chemical_symbols)
        nmetals = list(set(df.columns)-set(chemical_symbols))
        metals_remove = list(set(df.columns)-set(nmetals)-set(self.metals))
        self.df = self.df[~self.df.attributes.str.contains('Internal Standard|empty')]
        self.df = self.df.loc[((self.df[metals_remove]==0)|(self.df[metals_remove]==None)).all(axis = 1)]
        self.df.drop(columns = metals_remove, inplace = True)
        self.df.loc[len(self.df)] = 0
      
        #create dataframe with sklearn polynomial features
        self.df_path = self.df.copy()
        #self.df_path['path'] =  [j.split('/')[-1] for j in self.df.directory.values.tolist()]

        self.df = self.df.drop(columns = np.setdiff1d(self.df.columns, (self.comp+[self.target])))
        self.df_path = self.df_path.drop(columns = np.setdiff1d(self.df_path.columns, (self.comp+[self.target]+['directory'])))
          
        comps = []
        for i in self.comp:
          comps.append(self.df[i])
          self.X = np.column_stack(comps)

          self.X = PolynomialFeatures(2).fit_transform(self.X)

          self.y = self.df[self.target]
        
        model = sm.OLS(self.y,self.X).fit()
        self.model = model
    
      def model_sum(self):
        return self.model.summary()
      
      def plot_model(self):
        ''' Creates parity plot between true values and ols predicted values using the model  paramters
        '''
        pred = self.model.predict()
        plt.scatter(self.df[self.target], pred)
        plt.plot(np.linspace(0,self.df[self.target].max()+2), np.linspace(0,self.df[self.target].max()+2))
        plt.xlabel('True Value')
        plt.ylabel('Predicted Value')
        plt.title(f'{self.target} prediction and true values for {self.metals}')
        return plt
       
      def prediction(self):
        pred = self.model.predict()
        return pred
      
      def objective(self, X):
        ''' X is array of composition values
        Used with sklearn's minimize function to find optimal composition
        '''
        X= np.atleast_2d(X)
        Xp = PolynomialFeatures(2).fit_transform(X)
        return -self.model.predict(Xp)
    
      def optimum(self):
        ''' Finds optimum composition values 
        bounds are identified as minimum and maximum values in the data
        mid points are the guesses in the minimize function
        '''
        bounds = []
        mid = []
        for i in self.comp:
          if i == 'PS':
            min = self.df[i].min()
            max = 0.5
          else:
            min = self.df[i].min()
            max = self.df[i].max()
          bounds.append((min, max))
          mid.append((max+min)/2)
        return minimize(self.objective, mid, bounds=bounds) 

      def avg_df(self):
        '''
        '''
        unique_vals = []
        for i in self.comp:
          unique_vals.append(self.df[i].unique())
        df1 = pd.DataFrame(columns = self.comp + [f'{self.target}_avg', f'{self.target}_med', f'{self.target}_stddev'])
        if len(self.comp) == 3:
              for i in unique_vals[0]:
                for j in unique_vals[1]:
                  for k in unique_vals[2]:
                    df_temp = self.df[(self.df[self.comp[0]] == i)&(self.df[self.comp[1]] == j)&(self.df[self.comp[2]] == k)]
                    if len(df_temp)>0:
                      avg = np.array(df_temp[self.target].values.tolist()).mean(axis = 0)
                      med = np.median(np.array(df_temp[self.target].values.tolist()))
                      error = np.array(df_temp[self.target].values.tolist()).std(axis = 0)
                      index = df_temp.index.values
                      df1 = pd.concat(df1, pd.DataFrame({self.comp[0]: i, 
                                                self.comp[1]: j, 
                                                self.comp[2]: k, 
                                                f'{self.target}_avg': avg, 
                                                f'{self.target}_med': med, 
                                                f'{self.target}_stddev': error, 
                                                'wells': [index]}))
        if len(self.comp) == 2:
              for i in unique_vals[0]:
                for j in unique_vals[1]:
                    df_temp = self.df[(self.df[self.comp[0]] == i)&(self.df[self.comp[1]] == j)]
                    if len(df_temp)>0:
                      avg = np.array(df_temp[self.target].values.tolist()).mean(axis = 0)
                      error = np.array(df_temp[self.target].values.tolist()).std(axis = 0)
                      index = df_temp.index.values
                      df1 = pd.concat(df1, pd.DataFrame({self.comp[0]: i, 
                                                         self.comp[1]: j, 
                                                         f'{self.target}_avg': avg, 
                                                         f'{self.target}_stddev': error, 
                                                        'wells': [index]}))
        return df1
    
      def plot_surface_2d(self,bounds = [2,2]):
            '''plots surface for a 2 variable system
            Takes in bounds which is an array of max values for comp 1 and comp 2
            outputs a 3 d plot where the y axis and color is activity
            '''
            par = self.model.params.values
            x1 = np.linspace(0, bounds[0], 10) 
            x2 = np.linspace(0, bounds[1] , 10) 
            X1,X2 = np.meshgrid(x1, x2)
            X1 = X1.flatten()
            X2 = X2.flatten()
            def y(X1, X2, par):
                y = (par[0] + par[1]*X1 + par[2]*X2 + par[3]*X1**2+ 
                      par[4]*X1*X2 + par[5]*X2**2)
                return y

            Y = y(X1, X2, par)

            size = int(np.sqrt(len(X1)))
            X1 = X1.reshape(size,size)
            X2 = X2.reshape(size, size)
            Y = Y.reshape(size, size)


            fig = plt.figure(figsize = (8,6))
            ax = fig.gca(projection='3d')
            surf = ax.plot_surface(X1, X2, Y, rstride=1, cstride=1, 
                                    cmap=plt.cm.viridis,linewidth=0, antialiased=False)
            s = ''
            for i in range(len(self.comp)):
              s+= str(self.comp[i]) + ' '
            ax.set_title(f'{self.target} for'+s)
            ax.set_xlabel(comp[0]+' mM')
            ax.set_ylabel(comp[1]+' mM')
            ax.set_zlabel(self.target)
            ax.set_xticks(np.linspace(0,bounds[0], 5))
            ax.set_yticks(np.linspace(0,bounds[1], 5))
            fig.colorbar(surf, shrink=0.5, aspect=10)
            plt.show()
            
      def plot_surface(self, mbound = 2, psbound = 0.5, pegbound = 2):
        '''
        Plots the 3D surface of the metal and pegsh with a fixed PS concentration.
        mbound(int) is the metal surface plot axis limit
        psbound(int) is the fixed PS concentration
        pegbound(int) is the pegsh surface plot axis limit
        '''
        par = self.model.params.values
        x1 = np.linspace(0, mbound, 10) 
        x2 = np.linspace(0, 2 , 10) 
        x3 = np.linspace(0,pegbound, 10)
        X1,X2, X3 = np.meshgrid(x1, x2, x3)
        X1 = X1.flatten()
        X2 = X2.flatten()
        X3 = X3.flatten()
        def y(X1, X2, X3, par):
            y = (par[0] + par[1]*X1 + par[2]*X2 + par[3]*X3 + 
                 par[4]*X1**2 + par[5]*X1*X2 + par[6]*X1*X3 + 
                 par[7]*X2**2 + par[8]*X2*X3 + par[9]*X3**2)
            return y
        
        Y = y(X1, X2**0*psbound, X3, par)
        
        size = int(np.cbrt(len(X1)))
        X1 = X1[0:size**2].reshape(size,size)
        X2 = X2[0:size**2].reshape(size, size)
        X3 = X3[0:size**2].reshape(size, size)
        Y = Y[0:size**2].reshape(size, size)
        
        fig = plt.figure(figsize = (8,6))
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X1, X3, Y, rstride=1, cstride=1, 
                               cmap=plt.cm.viridis,linewidth=0, antialiased=False)
        
        ax.set_title(f'{self.target} for {self.metals} and PEGSH mM at PS={psbound}mM')
        ax.set_xlabel('metal_conc mM')
        ax.set_ylabel('PEGSH_conc mM')
        ax.set_zlabel(self.target)
        ax.set_xticks(np.linspace(0,mbound, 5))
        ax.set_yticks(np.linspace(0,pegbound, 5))
        fig.colorbar(surf, shrink=0.5, aspect=10)
        return plt.show()
        
      def plot_heatmap(self,mbound = 2, psbound = 0.5, pegbound = 2):
        '''
        Plots heatmap of the metal and pegsh with a fixed PS concentration.
        mbound(int) is the metal surface plot axis limit
        psbound(int) is the fixed PS concentration
        pegbound(int) is the pegsh surface plot axis limit
        '''
        par = self.model.params.values
        x1 = np.linspace(0, mbound, 10) 
        x2 = np.linspace(0, 2 , 10) 
        x3 = np.linspace(0,pegbound, 10)
        X1,X2, X3 = np.meshgrid(x1, x2, x3)
        X1 = X1.flatten()
        X2 = X2.flatten()
        X3 = X3.flatten()
        def y(X1, X2, X3, par):
            y = (par[0] + par[1]*X1 + par[2]*X2 + par[3]*X3 + 
                 par[4]*X1**2 + par[5]*X1*X2 + par[6]*X1*X3 + 
                 par[7]*X2**2 + par[8]*X2*X3 + par[9]*X3**2)
            return y
        
        Y = y(X1, X2**0*psbound, X3, par)
        
        size = int(np.cbrt(len(X1)))
        Y = Y[0:size**2].reshape(size, size)
        X1 = (X1[0::10][0:10])
        X2 = (X2[0::100])
      
        fig = go.Figure(go.Heatmap(z=Y, x = X1, y = X2,
                                   zmin=0, zmax=25,
                                   hovertemplate=str(self.metals[0])+ '(mM): %{x}<br>PEGSH(mM): %{y}<br>'+ str(self.target)+': %{z}<extra></extra>',
                                   hoverinfo='text'))
        fig.update_xaxes(title='Metal Conc (mM)')
        fig.update_yaxes(title='PEGSH Conc (mM)')
        fig.update_layout(title={'text': 'Max H<sub>2</sub> (umol) ' + str(self.metals[0])})
        return fig.show()
      
      def design_mat(self, design, dev):
        #Make mapping of concentrations
        opt = self.optimum().x
        map = np.empty((0,2), int)
        for i in range(len(opt)):
          min = opt[i]-dev
          max = opt[i]+dev
          if min < 0:
            min = 0
          if i == self.comp.index('PS'):
            if opt[i]+dev>0.5:
              max = 0.5
          map = np.concatenate((map, [[min, max]]), axis = 0) 
    
        #Map design Matrix
        if design == 'bb':
          self.des = bbdesign(3)
        elif design == 'cci':
          self.des = ccdesign(2, center=(3,3), alpha = 'r', face = 'cci')
          self.des = sm.add_constant(self.des)
          self.des[:, [1, 0]] = self.des[:, [0, 1]]
        self.des = (self.des + 1) * (map[:, 1] - map[:, 0]) / 2 + map[:, 0]
        return self.des
      
      #def input_df(self, v_tot = 440, m_stock = 5, lig_stock = 16.5, ps_stock = 2.75):
      #  ''' For a given metal, turn the design matrix into a dataframe with all relevant solutions. Stock solutions concentrations and total volumes are optional parameters.
      #  '''
      #  #turn design concentrations into volumes using stock solutions 
      #  m_vols = [x*v_tot/m_stock for x in self.des[:,0]]
      #  ps_vols = [x*v_tot/ps_stock for x in self.des[:,1]]
      #  lig_vols = [x*v_tot/lig_stock for x in self.des[:,2]]
      #  lig_vols = [7 if 0<i<7 else i for i in lig_vols]
      #  wat_vols = [(40-i) for i in lig_vols]
      #  teoa_vols = 20*np.ones(len(m_vols))
    
        #Make DataFrame
      #  df1 = pd.DataFrame({})
      #  df1[f'{self.metals[0]}_vol'] = m_vols
      #  df1['PEGSH_vol'] = lig_vols
      #  df1['PS_vol'] = ps_vols
      #  df1['Water_vol'] = wat_vols
      #  df1['TEOA_vol'] = 20
      #  df1['tot_vol'] = 440
      #  df1['DMSO_vol'] = df1.tot_vol - (df1[f'{self.metals[0]}_vol']+ df1.PEGSH_vol + df1.Water_vol + df1.PS_vol+ df1.TEOA_vol)   #calculate DMSO vol
      #  return df1


#class input_files:
#    def __init__(self):
        
  
