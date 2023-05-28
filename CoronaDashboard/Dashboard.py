"""Importing the required libraries"""

from bs4 import BeautifulSoup
import requests
import numpy as np
import pandas as pd

# import plotly.offline as pyo
import plotly.graph_objs as go

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

"""Building the functions required to scrape the website"""

# ---------------------------------------------------------------------------

def get_country_data(country_line):
    """
    This function formats a given input line parsed from an html page.

    Parameters:
        country_line : str
            it is a row table row, that contains the data.

    Returns:
        line : list
            A list containing all the useful information retrieved.
    """
    import re
    line = country_line.strip().split("\n")
    line.pop(0)
    for i, element in zip(range(len(line)), line):
        if re.search("[1-9]+", element):
            line[i] = float(''.join(line[i].strip('+').split(",")))
        else:
            pass

    return line[:-1]


def get_column_names(tr):
    """
    This function return a well formatted list for the column names.
    """
    line = tr.strip("\n#").strip().split("\n")
    line[12] += line[13]
    line.pop(14)
    line.pop(13)
    return line[1:-1]


def scrape_corona_data():
    """
    This function scrapes the data from the target website and returns a well formatted dict that contains information about every given country.
    """
    from collections import \
        defaultdict  # Importing the defaultdict model, that will be used to store the information while scraping the website
    countries_data = defaultdict(dict)
    coronameter = requests.get(
        "https://www.worldometers.info/coronavirus/")  # requesting the index page from the server, it is also where our information resides
    bscorona = BeautifulSoup(coronameter.text, "lxml")  # parsing the webpage to a beautifulsoup object.
    corona_table = bscorona.find("table",
                                 id="main_table_countries_today")  # selecting the table where our data is contained.
    print(corona_table.tr.text)
    column_names = get_column_names(corona_table.tr.text)
    print(column_names)
    for tr in corona_table.find_all("tr", {"style": ""})[2:-2]:
        line = get_country_data(tr.text)
        countries_data[line[0]] = dict(zip(column_names, line[1:]))
    return countries_data


def replace_nan(data):
    """
    This function replaces empty or N/A values with np.nan so that it can be easier to manipulate the data later on.
    """
    for col in data.columns:
        data[col].replace(["N/A", "", " "], np.nan, inplace=True)


def create_clean_dataframe(countries_data):
    """
    This function takes a dict object and create a clean well formatted dataframe.

    Parameters:
        countries_data : dict object
            The dict that contains the countries data.
    Returns:
        data : dataframe
            Well formatted dataframe.
    """
    data = pd.DataFrame(countries_data).transpose()
    replace_nan(data)
    return data


"""Building the plotting functions"""

# ---------------------------------------------------------------------------

def plot_continent_data(data, keyword):
    """
    This function creates a Figure from continental data.

    Parameters:
        data : dataframe
            The whole dataset.
        keyword : str
            The keyword used to define the figure wanted, the available keyword : {"Total", "New"}

    Returns:
        fig : Figure
            The figure that will be drawed on plotly.
    """
    if keyword == "New":
        cols = ["NewCases", "NewRecovered", "NewDeaths"]
    else:
        cols = ["TotalCases", "TotalRecovered", "TotalDeaths"]
    res = data.groupby("Continent")[cols].sum()

    plot_data = []
    colors = ["#031cfc", "#24b514", "#d11d1d"]
    for col, color in zip(cols, colors):
        plot_data.append(go.Bar(x=res.index.to_list(), y=res[col], name=col, marker=dict(color=color)))

    layout = go.Layout(title=f"Corona {keyword} Cases/Recovered/Deaths",
                       xaxis=dict(title="Continents"),
                       yaxis=dict(title="Cases per Continent"))

    fig = go.Figure(data=plot_data, layout=layout)
    return fig


def get_continent_sorted_data(data, continent, sortedby="TotalCases", ascending=False):
    """
    This function creates a sorted dataframe related to a continent and sorted by a columns.

    Parameters:
        data : dataframe
            The whole dataset.
        continent : str
            The continent we want to get the data from.
        sortedby : str, Default="TotalCases"
            The name of the column we want to sort by.
        ascending : Boolean, Default=False
            Either we want to sort in an ascending order or descending order.
    Returns:
        groupedbydata : dataframe
            A dataframe groupedby the continent.
    """
    return data.groupby("Continent").get_group(continent).sort_values(by=sortedby, ascending=ascending).reset_index()


def get_top_k_countries(data, k_countries=10, sortedby="TotalCases", ascending=False):
    """
    This function creates a k-len dataframe sorted by a key.

    Parameters:
        data : dataframe.
            The whole dataset.
        k_countries : int, Default=10
            The number of countries you want to plot.
        sortedby : str, Default="TotalCases".
            The column name we want to sort the data by
        ascending : Boolean, Default=False
            Either we want to sort in an ascending order or descending order.

    Returns:
        data : dataframe
            The k_contries lines dataframe sortedby the key given and in the wanted order.
    """
    return data.sort_values(by=sortedby, ascending=ascending).iloc[:k_countries]


def get_latitude_longitude(data):
    # Import the required library
    from geopy.geocoders import Nominatim

    # Initialize Nominatim API
    geolocator = Nominatim(user_agent="MyApp")

    lats=[]
    longs=[]
    for i in data.index[1:60]:
        location = geolocator.geocode(i)
        lats.append(location.latitude)
        longs.append(location.longitude)
    return lats, longs


def plot_top_k_countries(n_countries, sortby):
    """This function returns a figure where a number of countries are sorted by the value that resides in sortby."""
    res = get_top_k_countries(data, n_countries, sortby)
    plot_data = []

    plot_data.append(go.Bar(x=res.index.to_list(), y=res[sortby], name=sortby))

    layout = go.Layout(title=f"Top Countries orderedby {sortby}",
                       xaxis=dict(title="Countries"),
                       yaxis=dict(title=f"{sortby}"))

    fig = go.Figure(data=plot_data, layout=layout)
    return fig


def plot_boxplots(data, keyword="Deaths/1M pop"):
    """This function returns a figure of the boxplot related to each continent in regards to the keyword."""
    plot_data = []
    grouped_data = data.groupby("Continent")
    continents = data["Continent"].value_counts().index.to_list()
    for continent in continents:
        plot_data.append(go.Box(y=grouped_data.get_group(continent)[keyword], name=continent))
    layout = go.Layout(title=f"Boxplots using {keyword}",
                       xaxis=dict(title="Continents"),
                       yaxis=dict(title=f"{keyword}"))
    fig = go.Figure(data=plot_data, layout=layout)
    return fig

def plot_pie(data, keyword = "Total Cases"):
    # Create the pie chart figure
    labels = list(data[keyword].index[0:5])
    values = list(data[keyword][0:5])
    fig = go.Figure(data=go.Pie(labels=labels, values=values))

    # Set the layout properties
    fig.update_layout(title='Pie Chart Example')
    return fig


def plot_map(data, keyword="TotalCases"):

    import plotly.express as px
    lats,longs = get_latitude_longitude(data)  # get the latitutde and longitude of each location 
    fig = px.scatter_mapbox(data[1:60][:], lat=lats, lon=longs, hover_name=data.index[1:60], hover_data=["Continent", keyword],
                        color_discrete_sequence=["fuchsia"], zoom=3, height=500)
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    return fig

def plot_heatmap(data):
    # Calculate the correlation matrix
    # Select specific columns
    selected_columns = ['ActiveCases', 'TotalCases', 'TotalDeaths', 'TotalRecovered', 'TotalTests']

    # Create a new dataframe with selected columns
    data = pd.DataFrame(data, columns=selected_columns)
    corr_matrix = data.corr()

    # Create a heatmap using Plotly graph_objects
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='Viridis',
        reversescale=True,
        colorbar=dict(title='Correlation')
    ))

    # Add annotations to the heatmap
    for i in range(len(corr_matrix.index)):
        for j in range(len(corr_matrix.columns)):
            fig.add_annotation(
                x=corr_matrix.columns[j],
                y=corr_matrix.index[i],
                text="{}".format(corr_matrix.iloc[i, j]),
                showarrow=False,
                font=dict(color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black')
            )

    # Update layout
    fig.update_layout(
        title='Correlation Matrix Heatmap',
    )

    return fig



def init_figure():
    "This function initiate all the needed figure to start the app."
    return plot_continent_data(data, keyword="New"), plot_top_k_countries(10, "TotalCases"), plot_boxplots(data), plot_pie(data, "TotalCases"), plot_map(data, "TotalCases"), plot_heatmap(data)



"""Initiale Figures"""
# ---------------------------------------------------------------------------

countries_data = scrape_corona_data()
data = create_clean_dataframe(countries_data)

init_continent_fig, init_k_countries_plot, init_box_fig, init_pie, init_map, init_heatmap = init_figure()



"""Building the app"""
# ---------------------------------------------------------------------------

# Initializing the app
app = dash.Dash(__name__)
server = app.server

# Building the app layout
app.layout = html.Div([
    html.H1("Corona Tracker DashBoard", style={"text-align": "center"}),
    html.Br(),
    html.Div([
        html.Br(),
        html.H2("Corona Cases/Recovered/Deaths by Continent", style={"text-align": "center"}),
        html.Br(),
        dcc.Dropdown(id="select_keyword",
                     options=[
                         dict(label="Today's Data", value="New"),
                         dict(label="Total Data", value="Total")],
                     multi=False,
                     value="New",
                     style={"width": "40%"}
                     ),

        dcc.Graph(id="continent_corona_bar", figure=init_continent_fig)
    ]),

    html.Div([
        html.Br(),
        html.H2("Visualize Countries by attribute.", style={"text-align": "center"}),
        html.Br(),
        dcc.Dropdown(id="select_attribute",
                     options=[
                         dict(label="Total Cases", value='TotalCases'),
                         dict(label="New Cases", value='NewCases'),
                         dict(label="Total Cases per 1M population", value='Tot Cases/1M pop'),
                         dict(label="Active Cases", value='ActiveCases'),
                         dict(label="Serious, Critical Cases", value='Serious,Critical'),
                         dict(label="Total Deaths", value='TotalDeaths'),
                         dict(label="New Deaths", value='NewDeaths'),
                         dict(label="Deaths per 1M population", value='Deaths/1M pop'),
                         dict(label="Total Recovered", value='TotalRecovered'),
                         dict(label="New Recovered", value='NewRecovered'),
                         dict(label="Total Tests", value='TotalTests'),
                         dict(label="Tests per 1M population", value='Tests/1M pop')],
                     multi=False,
                     value="TotalCases",
                     style={"width": "60%", 'display': 'inline-block'}
                     ),
        dcc.Dropdown(id="select_k_countries",
                     options=[
                         dict(label="Top 5", value=5),
                         dict(label="Top 10", value=10),
                         dict(label="Top 25", value=25),
                         dict(label="Top 50", value=50),
                     ],
                     multi=False,
                     value=10,
                     style={"width": "30%", 'display': 'inline-block'}
                     ),

        dcc.Graph(id="k_countries_sorted", figure=init_k_countries_plot)
    ]),

    html.Div([
        html.Br(),
        html.H2("BoxPlot to explain the distribution of the variables", style={"text-align": "center"}),
        html.Br(),
        dcc.Dropdown(id="select_box_attribute",
                     options=[
                         dict(label="Deaths per 1M population", value='Deaths/1M pop'),
                         dict(label="Tests per 1M population", value='Tests/1M pop')
                     ],
                     multi=False,
                     value="Deaths/1M pop",
                     style={"width": "40%"}
                     ),

        dcc.Graph(id="continent_box_plot", figure=init_box_fig)
    ]),

    html.Div([
    html.Br(),
    html.H2("Pie Chart to compare the active and died cases", style={"text-align": "center"}),
    html.Br(),
    dcc.Dropdown(id="piechart_id",
                    options=[
                        dict(label="Cases per 1M population", value='TotalCases'),
                        dict(label="Total deaths per 1M population", value='TotalDeaths'),
                        dict(label="Active cases per 1M population", value='ActiveCases')

                    ],
                    multi=False,
                    value="Total Cases",
                    style={"width": "40%"}
                    ),

        dcc.Graph(id="piechart_active", figure=init_pie)
    ]),

    html.Div([
    html.Br(),
    html.H2("GeoLocalisation of coronavirus details in the map", style={"text-align": "center"}),
    html.Br(),

        dcc.Graph(id="map_active", figure=init_map)
    ]),

    html.Div([
    html.Br(),
    html.H2("Correlation Measurement of dataset variables", style={"text-align": "center"}),
    html.Br(),
        dcc.Graph(id="heatmap_active", figure=init_heatmap)
    ])


])


# Defining the application callbacks

@app.callback(
    Output("continent_corona_bar", "figure"),
    Input("select_keyword", "value")
)
def update_continent_corona_bar(value):
    return plot_continent_data(data, keyword=value)


@app.callback(
    Output("k_countries_sorted", "figure"),
    Input("select_attribute", "value"),
    Input("select_k_countries", "value")
)
def update_k_countries_sorted(value):
    return plot_top_k_countries(data, keyword=value)


@app.callback(
    Output("continent_box_plot", "figure"),
    Input("select_box_attribute", "value")
)
def update_continent_box_plot(value):
    return plot_boxplots(data, keyword=value)

@app.callback(
    Output("piechart_active", "figure"),
    Input("piechart_id", "value")
)
def update_piechart_plot(value):
    return plot_pie(data, keyword=value)

@app.callback(
    Output("map_active", "figure"),
    Input("map_id", "value")
)
def update_map_plot(value):
    return plot_map(data, keyword=value)

@app.callback(
    Output("heatmap_active", "figure"),
    #Input("map_id", "value")
)
def update_heatmap_plot():
    return plot_heatmap(data)


if __name__ == "__main__":
    countries_data = scrape_corona_data()
    data = create_clean_dataframe(countries_data)
    app.run_server()
