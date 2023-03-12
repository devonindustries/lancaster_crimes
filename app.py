# IMPORTS
# --------
import dash
from dash import dcc, html, Output, Input

import datetime
import math

import pandas as pd
import plotly.express as px

# APP SETTINGS
# --------
ext_style = [
    {"rel" : "stylesheet"}
]
app = dash.Dash(__name__, external_stylesheets=ext_style)
app.title = "Understanding crime statistics in the Lancaster Area."

# FUNCTIONS
# --------
def toDatetime(val): 
    return datetime.datetime.strptime(val, '%Y-%m')

def fromDatePicker(val):
    val = val.split('-')[:2]
    return toDatetime(f'{val[0]}-{val[1]}')

def yearToFloat(dtobj):
    return dtobj.year + (dtobj.month - 1) / 12

def floatToYear(float):
    year = int(math.floor(float))
    month = int((float - year) * 12) + 1
    return datetime.datetime(year, month, 1)

# DATA
# --------

data = pd.read_csv('Data/LancasterCrimesClean.csv')

lsoa_households = 650
lsoa_population = 1500

all_crimes = data['CT_simplified'].unique()
all_crimes.sort()

dates = [toDatetime(_) for _ in data.Month.unique()]
dates.sort()

# GRAPHS
# --------
def GeneratePieChart(data):
    
    # Data cleaning
    lsoa_counts = data.groupby('LSOA name').count().sort_values(by='LSOA code', ascending=False)['LSOA code'].reset_index()[:10]
    lsoa_counts['LSOA name'] = lsoa_counts['LSOA name'].map(lambda x : x.split()[-1])

    # Figure Generation
    fig = px.pie(
        lsoa_counts, 
        values="LSOA code", 
        names="LSOA name", 
        title="Highest 10 LSOAs by Crime Count"
    )

    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(showlegend=False)

    return fig

def GenerateBarChart(data):

    # Figure Generation
    crime_counts = data.groupby(['CT_simplified', 'Result']).count()['Month'].reset_index().sort_values(by='Month', ascending=True)

    fig = px.bar(
        crime_counts, 
        x="Month", 
        y="CT_simplified", 
        color="Result", 
        barmode="group", 
        color_discrete_map={
            'Unavailable':'Grey', 
            'Unresolved':'Red', 
            'Resolved':'Green'
        }, 
        labels={
            'CT_simplified' : '', 
            'Month' : 'Cases'
        },
        title="Distribution of each crime"
    )
    
    return fig

def GenerateTimeSeries(data):

    # Data Cleaning
    time_series_crimes = data.groupby(['Month', 'CT_simplified']).count()['LSOA code'].reset_index()

    # Figure Generation
    fig = px.line(
        time_series_crimes, 
        x='Month', 
        y='LSOA code', 
        color='CT_simplified',
        labels={
            'LSOA code' : 'Count',
            'CT_simplified' : 'Crime type'
        }
    )
    # fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99))

    return fig

def GenerateLSOAMap(data):

    # Take a look at the mean number of crimes for each LSOA
    lsoa_results = pd.concat([
        data.groupby('LSOA name')['Result'].count(), 
        data.groupby('LSOA name').mean()
    ], axis=1)

    lsoa_results = lsoa_results.reset_index()
    lsoa_results.columns = ['LSOA', 'Crimes', 'Longitude', 'Latitude']

    # Perform some further feature engineering
    lsoa_results['Crimes Per Household'] = lsoa_results['Crimes'] / lsoa_households
    lsoa_results['Crimes Per Person'] = lsoa_results['Crimes'] / lsoa_population

    lsoa_results[['Crimes Per Household', 'Crimes Per Person']] = lsoa_results[['Crimes Per Household', 'Crimes Per Person']].apply(lambda x : round(x, 3))

    # Figure Generation
    fig = px.scatter_mapbox(lsoa_results, lat="Latitude", lon="Longitude", color="Crimes", size='Crimes', hover_name="LSOA", hover_data=["Crimes", "Crimes Per Household", "Crimes Per Person"], zoom=12)
    fig.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})

    return fig

def GenerateIndividualMap(data):

    # Figure generation
    fig = px.scatter_mapbox(
        data, 
        lat="Latitude", 
        lon="Longitude", 
        color="Result", 
        hover_name="LSOA name", 
        hover_data=["CT_simplified", "Month"], 
        zoom=12, 
        color_discrete_sequence=['Blue', 'Red', 'Green'],
        labels={
            'CT_simplified' : 'Crime type'
        }
    )
    fig.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0}, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    
    return fig

def GenerateScatterMap(map_type, data):

    # Check which type of map we are returning
    if map_type=="Area Code":
        return GenerateLSOAMap(data)
    else:
        return GenerateIndividualMap(data)

# DATA
# --------
def GenerateStatistics(data):

    # TOTAL CASES
    total_cases = f'{len(data):,}'

    # RESOLUTION RATE
    results = data.groupby('Result').count()['CT_simplified']
    resolution_rate = f"{round(results['Resolved'] / (results['Resolved'] + results['Unresolved']) * 100, 2)}%"

    # BIGGEST CRIME
    crime_counts = data.groupby('CT_simplified').count()['Result']
    biggest_crime = crime_counts.reset_index().iloc[crime_counts.argmax()]

    # BIGGEST MONTH
    month_counts = data.groupby('Month').count()['Result']
    biggest_month = month_counts.reset_index().iloc[month_counts.argmax()]

    return([
        total_cases, 
        resolution_rate, 
        f'{biggest_crime[0]} with {biggest_crime[1]:,} cases.',
        f'{biggest_month[0]} with {biggest_month[1]:,} cases.'
    ])

# LAYOUT
# --------
app.layout = html.Div(children=[
    
    # HEADER
    # --------
    html.Header(children=[
        html.H1(children="Lancaster Crime Analytics", className="header-title"),
    ], className="header"),

    # MAIN BODY
    # --------
    html.Div(children=[
        
        # SIDEBAR
        # --------
        html.Aside(children=[

            # Title
            # --------
            html.Div(children=[
                html.H2("FILTERS")
            ], className="sidebar-title"),

            # Crime Type filter
            # --------
            html.H3("Crime Type:"),
            html.Div(children=[
                dcc.Checklist(options=[
                    {"label" : html.Label(crime), "value" : crime} for crime in all_crimes
                ], value=all_crimes, id="heatmap-checkbox", className="checkbox")
            ], className="scrollable"),

            # Year Range Slider
            # --------
            html.H3("Year Range:"),
            html.Div(children=[
                dcc.RangeSlider(yearToFloat(min(dates)), yearToFloat(max(dates)),
                    marks={i+1:str(i+1) for i in range(int(yearToFloat(min(dates))), int(yearToFloat(max(dates)))) if i==round(i)},
                    value=[yearToFloat(min(dates)), yearToFloat(max(dates))],
                    step=1/12,
                    vertical=True,
                    id="date-picker-range"
                )
            ], className="scrollable date-picker"),

            # OTHER STATS
            # --------
            html.H2("STATISTICS"),
            html.Div(children=[
                html.Div(children=[
                    html.H4("Total Cases:"),
                    html.H3(id="total-cases")
                ]),
                html.Div(children=[
                    html.H4("Resolution Rate:"),
                    html.H3(id="resolution-rate")
                ]),
                html.Div(children=[
                    html.H4("Biggest Crime:"),
                    html.H3(id="biggest-crime")
                ]),
                html.Div(children=[
                    html.H4("Biggest Month:"),
                    html.H3(id="biggest-month")    
                ])

            ], className="scrollable"),
            html.P("Note that the above statistics are calculated based on the filters above.")
            
        ], className="sidebar"),

        # MAIN CONTENT
        # --------
        html.Div(children=[
            
            # LEFT CARD
            # --------
            html.Section(children=[
                
                # PIECHART
                # --------
                dcc.Graph(id="piechart", className="graph"),

                # BARCHART
                # --------
                dcc.Graph(id="barchart", className="graph")
                
            ], className="card"),

            # RIGHT CARD
            # --------
            html.Section(children=[

                # HEATMAP
                # --------
                dcc.Graph(id="heatmap", className="graph map"),

                html.Div(children=[
                    html.Div("Map Type: "),
                    dcc.RadioItems(options=['Area Code', 'Individual'], value='Area Code', inline=True, id="heatmap-radio")
                ], className="heatmap-options graph")
                
            ], className="card"),

            # TIME SERIES
            # --------
            dcc.Graph(id="timeseries", className="graph-big")

        ], className="wrapper")

    ])
])

# CALLBACKS
# --------
@app.callback(
    # OUTPUTS
    [
        # GRAPHS
        # ------
        Output("piechart", "figure"),
        Output("barchart", "figure"),
        Output("timeseries", "figure"),
        Output("heatmap", "figure"),

        # RAW DATA
        # ------
        Output("total-cases", "children"),
        Output("resolution-rate", "children"),
        Output("biggest-crime", "children"),
        Output("biggest-month", "children")
        
    ],
    # INPUTS
    [
        Input("heatmap-radio", "value"), 
        Input("heatmap-checkbox", "value"),
        Input("date-picker-range", "value")
    ]
)
def RefreshGraphs(map_type, crime_filter, date_range):

    # Convert to list
    if type(crime_filter) is not list: 
        crime_filter = [crime_filter]

    # Convert the start and end dates
    start_date = floatToYear(date_range[0])
    end_date = floatToYear(date_range[1])

    # Define a filter
    filtered_data = data[
        (data['CT_simplified'].isin(crime_filter)) &
        (data['Month'].map(lambda _: toDatetime(_)) >= start_date) &
        (data['Month'].map(lambda _: toDatetime(_)) <= end_date)
    ]

    # Calculate some statistics first
    stats = GenerateStatistics(filtered_data)

    return [
        GeneratePieChart(filtered_data),
        GenerateBarChart(filtered_data),
        GenerateTimeSeries(filtered_data),
        GenerateScatterMap(map_type, filtered_data),
        
    ] + stats

# RUN APP
# --------
if __name__ == "__main__":
    app.run_server(debug=False)