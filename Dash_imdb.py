"""Importing the required libraries"""

import bs4
import requests
import time
import random as ran
import sys
import pandas as pd
import numpy as np
from requests import get
from warnings import warn
from time import sleep
from random import randint

# import plotly.offline as pyo
import plotly.graph_objs as go

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

"""Building the functions required to scrape the website"""

# ---------------------------------------------------------------------------

def get_top_scored_10():
    """
    This function scrapes the movies only with score equal to 10 by setting a condition on the rating variable

    Parameters:
        No parameters because each page has a specific url and different attributes for scraping

    Returns:
        movies_title : list
            A list containing all the movies titles.
        imdb_ratings : list
            A list containing all the movies scores which is obvious to equal 10.

    """
    url = "https://www.imdb.com/search/title/?release_date=2023-01-01,2023-12-31&sort=user_rating,desc"
    response = requests.get(url)
    soup = bs4.BeautifulSoup(response.content, "html.parser")

    movie_containers = soup.find_all("div", class_="lister-item mode-advanced")
    movies_title=[]
    imdb_ratings=[]


    # we iterate over each movie container
    for container in movie_containers:
        # I get the movie title and IMDb rating
        movie_title = container.h3.a.text
        imdb_rating = container.strong.text
        # Check if the rating is 10
        if imdb_rating == "10.0":
            movies_title.append(movie_title)
            imdb_ratings.append(imdb_rating)
    return movies_title, imdb_ratings


def scrape_box_office_chart(imdb_url):

    """
    This function scrapes the movies turnover details about the earnings in the weekends and the gross.

    Parameters:
        No parameters because each page has a specific url and different attributes for scraping

    Returns:
        box_office_data : list containing dictionaries for every movie gross amount and weekend amount

    """
    
    response = requests.get(imdb_url)

    # I create a BeautifulSoup object to parse the HTML content
    soup = bs4.BeautifulSoup(response.content, "html.parser")
    # The link that will be povided indicates a table that contains the box office chart
    chart_table = soup.find("table", class_="chart full-width")
    # Getting the rows in the table (excluding the header row)
    rows = chart_table.find_all("tr")[1:]
    box_office_data = []



    # I iterate over each row to extract the weekend, gross, and title information
    for row in rows:
        # Looking for the columns in each row
        columns = row.find_all("td")
        # Values Extraction
        title = columns[1].text.strip()
        weekend = columns[2].text.strip()
        gross = columns[3].text.strip()

        box_office_data.append({"Weekend": weekend, "Gross": gross, "Title": title})
    return box_office_data


def scrape_sci_fi_movies():
    pages = np.arange(1, 100, 50)   # we define the number of pages we want to scrape, the more we increase the number of pages the more it takes time to scrape
    headers = {'Accept-Language': 'en-US,en;q=0.8'} # If this is not specified, the default language is Chinese

    #I initialize empty lists to store the variables scraped
    titles = []
    years = []
    ratings = []
    genres = []
    runtimes = []
    imdb_ratings = []
    imdb_ratings_standardized = []
    metascores = []
    votes = []

    for page in pages:
        #get request for sci-fi
        response = get("https://www.imdb.com/search/title?genres=sci-fi&" 
                       + "start=" 
                       + str(page) 
                       + "&explore=title_type,genres&ref_=adv_prv", headers=headers)

        sleep(randint(8,15))
        #throw warning for status codes that are not 200
        if response.status_code != 200:
            warn('Request: {}; Status code: {}'.format(requests, response.status_code))
        page_html = bs4.BeautifulSoup(response.text, 'html.parser')

        movie_containers = page_html.find_all('div', class_ = 'lister-item mode-advanced')

        #extract the 50 movies for that page
        for container in movie_containers:
            #conditional for all with metascore
            if container.find('div', class_ = 'ratings-metascore') is not None:
                #title
                title = container.h3.a.text
                titles.append(title)

                if container.h3.find('span', class_= 'lister-item-year text-muted unbold') is not None:
                  #year released
                  year = container.h3.find('span', class_= 'lister-item-year text-muted unbold').text # remove the parentheses around the year and make it an integer
                  years.append(year)

                else:
                  years.append(None) # each of the additional if clauses are to handle type None data, replacing it with an empty string so the arrays are of the same length at the end of the scraping

                if container.p.find('span', class_ = 'certificate') is not None:

                  #rating
                  rating = container.p.find('span', class_= 'certificate').text
                  ratings.append(rating)

                else:
                  ratings.append("")

                if container.p.find('span', class_ = 'genre') is not None:

                  #genre
                  genre = container.p.find('span', class_ = 'genre').text.replace("\n", "").rstrip().split(',') # remove the whitespace character, strip, and split to create an array of genres
                  genres.append(genre)

                else:
                  genres.append("")

                if container.p.find('span', class_ = 'runtime') is not None:

                  #runtime
                  time = int(container.p.find('span', class_ = 'runtime').text.replace(" min", "")) # remove the minute word from the runtime and make it an integer
                  runtimes.append(time)

                else:
                  runtimes.append(None)

                if float(container.strong.text) is not None:

                  #IMDB ratings
                  imdb = float(container.strong.text) # non-standardized variable
                  imdb_ratings.append(imdb)

                else:
                  imdb_ratings.append(None)

                if container.find('span', class_ = 'metascore').text is not None:

                  #Metascore
                  m_score = int(container.find('span', class_ = 'metascore').text) # make it an integer
                  metascores.append(m_score)

                else:
                  metascores.append(None)

                if container.find('span', attrs = {'name':'nv'})['data-value'] is not None:

                  #Number of votes
                  vote = int(container.find('span', attrs = {'name':'nv'})['data-value'])
                  votes.append(vote)

                else:
                    votes.append(None)

    sci_fi_df = pd.DataFrame({'movie': titles,
                           'year': years,
                           'rating': ratings,
                           'genre': genres,
                           'runtime_min': runtimes,
                           'imdb': imdb_ratings,
                           'metascore': metascores,
                           'votes': votes}
                           )

    sci_fi_df.loc[:, 'year'] = sci_fi_df['year'].str[-5:-1] 

    sci_fi_df['n_imdb'] = sci_fi_df['imdb'] * 10
    final_df = sci_fi_df.loc[sci_fi_df['year'] != 'ovie'] 
    final_df.loc[:, 'year'] = pd.to_numeric(final_df['year'])

    return final_df







"""Building the plotting functions"""

# ---------------------------------------------------------------------------

####            Plot the top scored films
def plot_scored10_movies():

    # Create a table using Plotly graph_objects
    fig = go.Figure(data=[go.Table(
        header=dict(values=['Movie title', 'rating']),
        cells=dict(values=[get_top_scored_10()[0], get_top_scored_10()[1]])
    )])

    # Update layout
    fig.update_layout(
        title='Movie title and Score',
    )
    return fig



####            Plot  the box office data and compare between the distrubtion of earning in weekends and the total profit
def plot_box_office():
    imdb_url = "https://www.imdb.com/chart/boxoffice/?ref_=nv_ch_cht"

    # apply the implemented function
    box_office_data = scrape_box_office_chart(imdb_url)
    df = pd.DataFrame(box_office_data)
    df = df[["Weekend", "Gross", "Title"]]
    Weekends=[]
    grosses=[]
    for i in df["Weekend"]:
        Weekends.append(float(i[1:-1]) * 1000000)
    for j in df["Gross"]:
        grosses.append(float(j[1:-1]) * 1000000)
    # Create a scatter plot using Plotly graph_objects
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=Weekends,
        y=grosses,
        mode='markers',
        marker=dict(size=10),
        name='Weekend vs Gross'
    ))

    # Update layout
    fig.update_layout(
        title='Comparison of Gross and Weekend Amounts',
        xaxis_title='Weekend Amount',
        yaxis_title='Gross Amount'
    )

    return fig


####       Plot the summary table of all extracted variables and get insight about every column 

def plot_summary(data):
    # Get the summary statistics using describe() method
    summary_stats = data.describe().reset_index()

    # Create a table using Plotly graph_objects
    fig = go.Figure(data=[go.Table(
        header=dict(values=summary_stats.columns),
        cells=dict(values=[summary_stats[col] for col in summary_stats.columns])
    )])

    # Update layout
    fig.update_layout(
        title='Summary Statistics',
    )
    
    return fig

####      The heatmap to measure the correlation between all the varaibles to discover if a variable is impacted by the other.

def plot_heatmap(final_df):
    # Calculate the correlation matrix
    corr_matrix = final_df.corr()

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

####       Plotting the variables votes and rating, to see how is the distribution of votes over the rating variable

def plot_votes_rating(final_df):
    # Create a scatter plot using Plotly graph_objects
    fig = go.Figure(data=go.Scatter(
        x=final_df['n_imdb'],
        y=final_df['votes'],
        mode='markers',
        marker=dict(
            size=8,
            color='blue',
            opacity=0.5
        )
    ))

    # Update layout
    fig.update_layout(
        title='Number of Votes vs. IMDB Rating',
        xaxis=dict(title='IMDB Rating Standardized'),
        yaxis=dict(title='Number of Votes')
    )
    return fig

####      Plotting the average metascore per year

def plot_score_year(final_df):
    # Calculate the average metascores by year
    avg_metascores = final_df.groupby('year')['metascore'].mean()

    # Create a bar plot using Plotly graph_objects
    fig = go.Figure(data=go.Bar(
        x=avg_metascores.index,
        y=avg_metascores.values,
        marker_color='blue'
    ))

    # Update layout
    fig.update_layout(
        title='Avg. Metascore by Year',
        xaxis=dict(title='Year', tickangle=90),
        yaxis=dict(title='Avg. Metascore')
    )

    return fig




df_final = scrape_sci_fi_movies()

def init_figure():
    "This function initiate all the needed figure to start the app"
    return plot_scored10_movies(), plot_box_office(), plot_summary(df_final), plot_heatmap(df_final),plot_votes_rating(df_final), plot_score_year(df_final)


"""Initiale Figures"""
# ---------------------------------------------------------------------------


init_score10, init_box, init_sci_fi, init_heatmap, init_votes_rating, init_score_year = init_figure()


"""Building the app"""
# ---------------------------------------------------------------------------

# Initializing the app
app = dash.Dash(__name__)
server = app.server

# Building the app layout
app.layout = html.Div([
    html.H1("IMDB platform Tracker DashBoard", style={"text-align": "center"}),
    html.Br(),
    html.Div([
        html.Br(),
        html.H2("All movies rated 10", style={"text-align": "center"}),
        html.Br(),

        dcc.Graph(id="score10", figure=init_score10)
    ]),

    html.Div([
        html.Br(),
        html.H2("Movies earnings", style={"text-align": "center"}),
        html.Br(),
        html.Blockquote(
            children=[
                html.P("Weekend: Refers to the specific period from Friday to Sunday (or sometimes Thursday to Sunday) during which the box office earnings of a movie are typically measured. It represents the revenue generated by a movie over the course of a weekend."),
                html.P("Gross: Refers to the total amount of money that a movie has earned at the box office. It includes all the revenue generated by the movie since its release, not just the earnings from a specific weekend.")
            ]
        ),

        dcc.Graph(id="box", figure=init_box)
    ]),





    html.Div([
        html.Br(),
        html.H2("Details about every scraped attribute", style={"text-align": "center"}),
        html.Br(),

        dcc.Graph(id="sci_fi", figure=init_sci_fi)
    ]),






    html.Div([
    html.Br(),
    html.H2("Heatmap to measure the correlation  between all variables", style={"text-align": "center"}),
    html.Br(),

        dcc.Graph(id="heatmap_active", figure=init_heatmap)
    ]),

    html.Div([
    html.Br(),
    html.H2("Does a well scored movie get more votes ?", style={"text-align": "center"}),
    html.Br(),

        dcc.Graph(id="votes_rating", figure=init_votes_rating)
    ]),


    html.Div([
    html.Br(),
    html.H2("Movies Average score per year", style={"text-align": "center"}),
    html.Br(),

        dcc.Graph(id="score_year", figure=init_score_year)
    ])


])


# Defining the application callbacks

@app.callback(
    Output("score10", "figure"),
    #Input("select_keyword", "value")
)
def update_score10_plot():
    return plot_scored10_movies()

@app.callback(
    Output("box", "figure"),
    #Input("select_keyword", "value")
)
def update_box_plot():
    return plot_box_office()

@app.callback(
    Output("sci_fi", "figure"),
    #Input("select_attribute", "value")
)
def update_sci_fi_plot():
    return plot_summary(df_final)

@app.callback(
    Output("heatmap_active", "figure"),
    #Input("select_attribute", "value")
)
def update_heatmap_plot():
    return plot_heatmap(df_final)

@app.callback(
    Output("votes_rating", "figure"),
    #Input("select_attribute", "value")
)
def update_votes_plot():
    return plot_votes_rating(df_final)

@app.callback(
    Output("score_year", "figure"),
    #Input("select_attribute", "value")
)
def update_votes_plot():
    return plot_score_year(df_final)



if __name__ == "__main__":
    df_final = scrape_sci_fi_movies()
    #data = create_clean_dataframe(countries_data)
    app.run_server()
