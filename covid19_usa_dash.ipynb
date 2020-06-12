# USA DASH
import numpy as np
import pandas as pd
import io
import requests
import plotly as py
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import datetime as dt
import math
import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from pylab import *

class CovidUS:
    def __init__(self):
        self.counties_url = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv'
        self.states_url = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv'
        self.counties_df = self.fetch_data(self.counties_url)
        self.counties_df['county'] = self.counties_df[['county', 'state']].agg(', '.join, axis=1)
        self.states_df = self.fetch_data(self.states_url)
        self.avg_window = 1
        self.num_locations = 10
        d1 = dt.date.today()
        y1 = d1 - dt.timedelta(days=2)
        self.yesterday = y1.strftime("%Y-%m-%d")
        self.start_date = '2020-02-27'
        # fips (Federal Information Processing Standards) used to identify counties (kinda like zip codes)
        county_names = self.counties_df.loc[:, ['county', 'fips']]
        #print(county_names.head())
        self.county_names = county_names.drop_duplicates()
        state_names = self.states_df.loc[:,['state','fips']]
        #print(self.states_df.head())
        #print(state_names.head())
        self.state_names = state_names.drop_duplicates()
        pop_df = pd.read_csv('2014_usa_states.csv')
        pop_df = pop_df.drop('Rank', axis=1)
        pop_df = pop_df.drop('Postal', axis=1)
        self.pop_df = pop_df.rename(columns={'Population': 'pop'})
        self.cases_per_cap = False
        self.deaths_per_cap = False

    def fetch_data(self, url):
        s = requests.get(url).content
        df = pd.read_csv(io.StringIO(s.decode('utf-8')))
        return df

    def set_locations_to_show(self, num_locations):
        self.num_locations = num_locations

    def set_start_date(self, start_date):
        self.start_date = start_date
        
    def set_avg_window(self, num_days):
        # set number of days for moving average
        self.avg_window=num_days

    def get_cases(self, option):
        if option == 'state':
            df = self.states_df.drop('deaths', axis=1)
        elif option == 'state per cap':
            df = self.states_df.drop('deaths', axis=1)
            df = df.merge(self.pop_df, left_on='state', right_on='State').drop('State',axis=1)
            df['cases_per_100k'] = ((df['cases']/df['pop'])*100000).round(5)
            df = df.drop('pop', axis=1)
            df = df.drop('cases', axis=1)
            df = df.rename(columns={'cases_per_100k':'cases'})
            option = 'state'
            self.cases_per_cap = True
        elif option == 'county':
            df = self.counties_df.drop('deaths', axis=1)
            df = df.drop('state', axis=1)
        df['date'] = pd.to_datetime(df['date'], format="%Y-%m-%d")
        df = df.groupby([option, 'date'])['cases'].sum().reset_index().sort_values('date', ascending=True)
        return df

    def get_deaths(self, option):
        if option == 'state':
            df = self.states_df.drop('cases', axis=1)
        elif option == 'state per cap':
            df = self.states_df.drop('cases', axis=1)
            df = df.merge(self.pop_df, left_on='state', right_on='State').drop('State',axis=1)
            df['deaths_per_100k'] = (df['deaths']/df['pop'])*100000
            df = df.drop('pop', axis=1)
            df = df.drop('deaths', axis=1)
            df = df.rename(columns={'deaths_per_100k':'deaths'})
            option = 'state'
            self.deaths_per_cap = True
        elif option == 'county':
            df = self.counties_df.drop('cases', axis=1)
            df = df.drop('state', axis=1)
        df['date'] = pd.to_datetime(df['date'], format="%Y-%m-%d")
        df = df.groupby([option, 'date'])['deaths'].sum().reset_index().sort_values('date', ascending=True)
        return df

    def get_new(self, df):
        # get new cases / new deaths by differentiating dataframe
        location = df.columns[0]
        if df.columns[2] == 'fips':
            variable = df.columns[3]
            df = df.groupby([location, 'date', 'fips'])[variable].sum().diff().reset_index()
        else:
            variable = df.columns[2]
            df = df.groupby([location, 'date'])[variable].sum().diff().reset_index()
        df.loc[df[variable] < 0, variable] = 0
        return df

    def filter_by(self, df, num_locations, after):
        location = df.columns[0]
        variable = df.columns[3]
        top = df[df['date'] == self.yesterday].sort_values(variable, ascending=False)
        top_list = top[location].unique()[0:num_locations].tolist()
        df = df[df[location].str.contains('|'.join(top_list), na=False)]
        df = df[df['date'] > after]
        df = df.groupby([location, 'date', 'fips'])[variable].sum().reset_index().sort_values('date', ascending=True)
        return df

    def get_hover_text(self, df):
        hover_text = []
        place = df.columns[0]
        place_id = df.columns[2]
        if len(df.columns) > 4:
            new = df.columns[3]
            variable = df.columns[4]
            for index, row in df.iterrows():
                hover_text.append((f'{place}: {{location}}<br>'+'Fips: {fips}<br>'+'Date: {date}<br>' +
                               f'{variable}: {{confirmed}}<br>'+'Increase: {increase}').format(location=row[place], 
                                date=row['date'], fips=row[place_id], confirmed=row[variable], increase=row[new]))    
        else:
            variable = df.columns[3]
            for index, row in df.iterrows():
                hover_text.append((f'{place}: {{location}}<br>'+'Fips: {fips}<br>'+'Date: {date}<br>' +
                                   f'{variable}: {{confirmed}}').format(location=row[place], date=row['date'],
                                    fips=row[place_id], confirmed=row[variable]))
        return hover_text

    def df_to_dict(self, df):
        place = df.columns[0]
        var_names = df[place].unique().tolist()
        var_data = {var: df.query(f'{place} == "%s"' % var) for var in var_names}
        return var_names, var_data

    def rolling_average(self, df):
        location = df.columns[0]
        variable = df.columns[2]
        df = df.pivot(index='date', columns=location, values=variable).fillna(0).reset_index()
        df.columns.name = None
        df.iloc[:, 1:] = df.iloc[:, 1:].rolling(window=self.avg_window).mean().round().fillna(0)
        df = df.melt(id_vars='date')
        df = df.rename(columns={'variable': location, 'value': variable})
        df = df[[location, 'date', variable]]
        return df

    def add_location_id(self, df):
        location = df.columns[0]
        variable = df.columns[2]
        if location == 'county':
            df = df.join(self.county_names.set_index(location), on=location)
        elif location == 'state':
            df = df.join(self.state_names.set_index(location), on=location)
        df = df[[location, 'date', 'fips', variable]].fillna(11111)
        return df
    
    def get_cases_dict(self, loc):
        self.avg_window = 7
        
        # cases = self.get_cases(option=loc)
        # cases = self.add_location_id(cases)
        # new_cases = self.get_new(cases)
    
        cases = self.get_cases(option=loc)
        new_cases = self.get_new(cases)
        
        # cases = self.rolling_average(cases)
        cases = self.add_location_id(cases)

        new_cases = self.rolling_average(new_cases)
        new_cases = self.add_location_id(new_cases)

        cases = self.filter_by(cases, num_locations=self.num_locations, after=self.start_date)
        cases_hover_text = self.get_hover_text(cases)
        cases['text'] = cases_hover_text
        cases_names, cases_data = self.df_to_dict(cases)
        
        cmap = cm.get_cmap('tab20', len(cases_names)) 
        cases_color_list = np.zeros(len(cases_names)).tolist()
        for i in range(cmap.N):
            rgb = cmap(i)[:3] # will return rgba, we take only first 3 so we get rgb
            cases_color_list[i] = matplotlib.colors.rgb2hex(rgb)
        cases_colors = dict(zip(cases_names, cases_color_list)) 
        
        new_cases = self.filter_by(new_cases, num_locations=self.num_locations, after=self.start_date)
        new_cases_hover_text = self.get_hover_text(new_cases)
        new_cases['text'] = new_cases_hover_text
        new_cases_names, new_cases_data = self.df_to_dict(new_cases)
        
        cmap = cm.get_cmap('tab20', len(new_cases_names)) 
        new_cases_color_list = np.zeros(len(new_cases_names)).tolist()
        for i in range(cmap.N):
            rgb = cmap(i)[:3] # will return rgba, we take only first 3 so we get rgb
            new_cases_color_list[i] = matplotlib.colors.rgb2hex(rgb)
        new_cases_colors = dict(zip(new_cases_names, new_cases_color_list)) 
        
        return cases_names, cases_data, cases_colors, new_cases_names, new_cases_data, new_cases_colors
    
    def get_deaths_dict(self, loc):
        self.avg_window = 7
        
        # deaths = self.get_deaths(option=loc)
        # deaths = self.add_location_id(deaths)
        # new_deaths = self.get_new(deaths)
        
        deaths = self.get_deaths(option=loc)
        new_deaths = self.get_new(deaths)
        
        # deaths = self.rolling_average(deaths)
        deaths = self.add_location_id(deaths)

        new_deaths = self.rolling_average(new_deaths)
        new_deaths = self.add_location_id(new_deaths)
        
        
        deaths = self.filter_by(deaths, num_locations=self.num_locations, after=self.start_date)
        deaths_hover_text = self.get_hover_text(deaths)
        deaths['text'] = deaths_hover_text
        deaths_names, deaths_data = self.df_to_dict(deaths)
        
        cmap = cm.get_cmap('tab20', len(deaths_names)) 
        deaths_color_list = np.zeros(len(deaths_names)).tolist()
        for i in range(cmap.N):
            rgb = cmap(i)[:3] # will return rgba, we take only first 3 so we get rgb
            deaths_color_list[i] = matplotlib.colors.rgb2hex(rgb)
        deaths_colors = dict(zip(deaths_names, deaths_color_list)) 
        
        new_deaths = self.filter_by(new_deaths, num_locations=self.num_locations, after=self.start_date)
        new_deaths_hover_text = self.get_hover_text(new_deaths)
        new_deaths['text'] = new_deaths_hover_text
        new_deaths_names, new_deaths_data = self.df_to_dict(new_deaths)
        
        cmap = cm.get_cmap('tab20', len(new_deaths_names)) 
        new_deaths_color_list = np.zeros(len(new_deaths_names)).tolist()
        for i in range(cmap.N):
            rgb = cmap(i)[:3] # will return rgba, we take only first 3 so we get rgb
            new_deaths_color_list[i] = matplotlib.colors.rgb2hex(rgb)
        new_deaths_colors = dict(zip(new_deaths_names, new_deaths_color_list)) 
        
        return deaths_names, deaths_data, deaths_colors, new_deaths_names, new_deaths_data, new_deaths_colors
    
    def get_full_cases_dict(self, loc):
        self.avg_window = 7

        # COMBINED
        cases = self.get_cases(option=loc)
        new_cases = self.get_new(cases)
        
        cases = self.rolling_average(cases)
        cases = self.add_location_id(cases)
        cases = self.filter_by(cases, num_locations=self.num_locations, after=self.start_date)

        new_cases = self.rolling_average(new_cases)
        new_cases = self.add_location_id(new_cases)
        new_cases = self.filter_by(new_cases, num_locations=self.num_locations, after=self.start_date)

        if self.cases_per_cap == True:
            # KEEP ONLY CASES / INCREASES OVER 3 PER 100K
            cases['cases'] = cases['cases'].mask(cases['cases'] < 10, np.nan)
            new_cases['cases'] = new_cases['cases'].mask(new_cases['cases'] < 1, np.nan)
        else:
            # KEEP ONLY CASES / INCREASES OVER 50
            cases['cases'] = cases['cases'].mask(cases['cases'] < 50, np.nan)
            new_cases['cases'] = new_cases['cases'].mask(new_cases['cases'] < 50, np.nan)

        # COMBINE DATA FRAMES, PUT INTO DICTIONARY
        cases_combined = cases.copy()
        cases_combined.insert(3, 'increase', new_cases.cases)
        combined_hover_text = self.get_hover_text(cases_combined)
        cases_combined['text'] = combined_hover_text
        cases_combined_names, cases_combined_data = self.df_to_dict(cases_combined)
        
        # GET COLOR MAP
        cmap = cm.get_cmap('tab20', len(cases_combined_names)) 
        combined_color_list = np.zeros(len(cases_combined_names)).tolist()
        for i in range(cmap.N):
            rgb = cmap(i)[:3] # will return rgba, we take only first 3 so we get rgb
            combined_color_list[i] = matplotlib.colors.rgb2hex(rgb)
        combined_cases_colors = dict(zip(cases_combined_names, combined_color_list)) 
        
        return cases_combined_names, cases_combined_data, combined_cases_colors


# UNITED STATES
covidus = CovidUS()
covidus.set_locations_to_show(num_locations=7)
covidus.set_start_date(start_date='2020-03-09')

# df = covidus.get_cases(option='state')
# df = covidus.get_new(df)
# df[df.state == 'North Carolina']
# CASES AND NEW CASES BY COUNTY
cases_names,cases_data,c_colors,new_cases_names,new_cases_data,c_colors_n = covidus.get_cases_dict(loc='state')
# DEATHS AND NEW DEATHS BY COUNTY
deaths_names,deaths_data,d_colors,new_deaths_names,new_deaths_data,d_colors_n = covidus.get_deaths_dict(loc='state')
# COMBINED
covidus.set_locations_to_show(num_locations=20)
cases_combined_names, cases_combined_data, cc_colors = covidus.get_full_cases_dict(loc='state')


cam_scale = 1.1
camera = dict(
    up=dict(x=0, y=0, z=1),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=cam_scale*2, y=cam_scale*0.1, z=cam_scale*0.5)
)

external_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

colors_dict = {'background': 'rgb(243, 243, 243)', 'text': 'black'}
w_scale = 1.4
h_scale = 1.4
grid_scale = 1.7


app.layout = html.Div([
    html.Hr(),
    dbc.Row(dbc.Col(html.H1(children='United States', style={'textAlign': 'center', 'color': colors_dict['text']}))),
    html.Hr(),
    dbc.Row([
            dbc.Col(html.Div(
                dcc.Graph(
                    id='us-cases',
                    figure=go.Figure(
                        data=[go.Scatter(
                            x=var['date'], 
                            y=var['cases'], 
                            text=var['text'], 
                            name=var_name, 
                            mode='lines+markers',
                            marker={'size': 5, 'color': c_colors[var_name], 
                                    'line': {'width': 0.5, 'color': c_colors[var_name]}},
                            line={'width':2, 'color': c_colors[var_name]},
                            ) for var_name, var in cases_data.items()],
                        layout=go.Layout(
                            title=dict(text="Cumulative Cases", x=0.5, font=dict(size=20)),
                            xaxis=dict(
                                title='Date', 
                                gridcolor='white', 
                                gridwidth=2, 
                                nticks=12, 
                                tickangle=-30,
                                rangeslider_visible=True
                            ),
                            yaxis=dict(title='Cases', gridcolor='white', gridwidth=2),
                            legend=dict(orientation="v", x=0.04, y=1.02, bordercolor="Grey", borderwidth=1),
                            margin=dict(l=150, r=40, b=100, t=100, pad=4),
                            height=grid_scale*500,
                            paper_bgcolor='rgb(243, 243, 243)', plot_bgcolor='rgb(243, 243, 243)'))
                    )),width=6),
            dbc.Col(html.Div(
                dcc.Graph(
                    id='us-deaths',
                    figure=go.Figure(
                        data=[go.Scatter(
                        x=var['date'], y=var['deaths'], text=var['text'], name=var_name, mode='lines+markers',
                        marker={'size': 5, 'color': d_colors[var_name], 
                                'line': {'width': 0.5, 'color': d_colors[var_name]}},
                        line={'width':2, 'color': d_colors[var_name]},
                        ) for var_name, var in deaths_data.items()],
                    layout=go.Layout(
                        title=dict(text="Cumulative Deaths", x=0.5, font=dict(size=20)),
                        xaxis=dict(
                            title='Date', 
                            gridcolor='white', 
                            gridwidth=2, 
                            nticks=12, 
                            tickangle=-30,
                            rangeslider_visible=True
                        ),
                        yaxis=dict(title='Deaths', gridcolor='white', gridwidth=2),
                        legend=dict(orientation="v", x=0.04, y=1.02, bordercolor="Grey", borderwidth=1),
                        margin=dict(l=40, r=150, b=100, t=100, pad=4),
                        height=grid_scale*500,
                        paper_bgcolor='rgb(243, 243, 243)', plot_bgcolor='rgb(243, 243, 243)'))
                    )),
                    width=6)]),
    dbc.Row([
            dbc.Col(html.Div(
                dcc.Graph(
                    id='us-new-cases',
                    figure=go.Figure(
                        data=[go.Scatter(
                        x=var['date'], y=var['cases'], text=var['text'], name=var_name, mode='lines+markers',
                        marker={'size': 5, 'color': c_colors_n[var_name], 
                                'line': {'width': 0.5, 'color': c_colors_n[var_name]}},
                        line={'width':2, 'color': c_colors_n[var_name]},
                        ) for var_name, var in new_cases_data.items()],
                    layout=go.Layout(
                        title=dict(text="New Cases", x=0.5, font=dict(size=20)),
                        xaxis=dict(
                            title='Date', 
                            gridcolor='white', 
                            gridwidth=2, 
                            nticks=12, 
                            tickangle=-30,
                            rangeslider_visible=True
                        ),
                        yaxis=dict(title='Cases', gridcolor='white', gridwidth=2),
                        legend=dict(orientation="v", x=0.04, y=1.02, bordercolor="Grey", borderwidth=1),
                        margin=dict(l=150, r=40, b=100, t=100, pad=4),
                        height=grid_scale*500,
                        paper_bgcolor='rgb(243, 243, 243)', plot_bgcolor='rgb(243, 243, 243)'))
                    )),
                    width=6),
            dbc.Col(html.Div(
                dcc.Graph(
                    id='us-new-deaths',
                    figure=go.Figure(
                        data=[go.Scatter(
                            x=var['date'], y=var['deaths'], text=var['text'], name=var_name, mode='lines+markers',
                            marker={'size': 5, 'color': d_colors_n[var_name], 
                                    'line': {'width': 0.5, 'color': d_colors_n[var_name]}},
                            line={'width':2, 'color': d_colors_n[var_name]},
                            ) for var_name, var in new_deaths_data.items()],
                        layout=go.Layout(
                            title=dict(text="New Deaths", x=0.5, font=dict(size=20)),
                            xaxis=dict(
                                title='Date', 
                                gridcolor='white', 
                                gridwidth=2, 
                                nticks=12, 
                                tickangle=-30,
                                rangeslider_visible=True
                            ),
                            yaxis=dict(title='Deaths', gridcolor='white', gridwidth=2),
                            legend=dict(orientation="v", x=0.04, y=1.02, bordercolor="Grey", borderwidth=1),
                            margin=dict(l=40, r=150, b=100, t=100, pad=4),
                            height=grid_scale*500,
                            paper_bgcolor='rgb(243, 243, 243)', plot_bgcolor='rgb(243, 243, 243)'))
                    )),
                    width=6),
            ], no_gutters=True),
    dbc.Row(
        dbc.Col(html.Div(
            dcc.Graph(
                id='new-cases-all',
                figure=go.Figure(
                    data=[go.Bar(
                        x=var['date'], y=var['cases'], text=var['text'], name=var_name,
                        marker={'color': c_colors_n[var_name]},
                        ) for var_name, var in new_cases_data.items()],
                    layout=go.Layout(
                        title=dict(text="Daily Increase in Cases", x=0.5, font=dict(size=20)),
                        barmode='stack',
                        xaxis=dict(
                            title='Date', 
                            gridcolor='white', 
                            gridwidth=2, 
                            nticks=12, 
                            tickangle=-30,
                            rangeslider_visible=True
                        ),
                        yaxis=dict(title='New Cases', gridcolor='white', gridwidth=2),
                        height=900,
                        legend=dict(orientation="v", x=0.04, y=1.02, bordercolor="Grey", borderwidth=1),
                        margin=dict(l=200, r=200, b=100, t=100, pad=4),
                        paper_bgcolor='rgb(243, 243, 243)', plot_bgcolor='rgb(243, 243, 243)')))),
                width={"size": 12})),
    dbc.Row([
        dbc.Col(html.Div(
            dcc.Graph(
                id='us-cases-increase-log',
                figure=go.Figure(
                    data=[go.Scatter3d(
                            x=var['date'], y=var['cases'], z=var['increase'], text=var['text'],
                            name=var_name, mode='lines', line={'width':7, 'color': cc_colors[var_name]},
                            ) for var_name, var in cases_combined_data.items()],
                    layout=go.Layout(
                        title=dict(text="Epidemic Case Trajectory", x=0.5, font=dict(size=20)),
                        scene=dict(xaxis_title=' <b>Date</b>', xaxis_gridwidth=2, xaxis_showspikes=False,
                                yaxis_title=' <b>Cases</b>', yaxis_gridwidth=2, yaxis_showspikes=False,
                                yaxis_type='log',
                                zaxis_title=' <b>Daily Increase</b>', zaxis_gridwidth=2,
                                zaxis_showspikes=False, zaxis_type='log'),
                        scene_camera=camera,
                        legend=dict(orientation="v", yanchor='top', bordercolor="Grey", borderwidth=1),
                        height=h_scale*1000,
                        margin=dict(l=100, r=100, b=50, t=100, pad=4),
                        paper_bgcolor='rgb(243, 243, 243)')))),
                width={'size': 10, 'offset': 1})]),
], style={'backgroundColor': colors_dict['background']})


if __name__ == '__main__':
    app.run_server(debug=False)

