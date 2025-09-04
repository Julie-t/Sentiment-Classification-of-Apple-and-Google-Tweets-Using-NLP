import os
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from flask import Flask, request, jsonify
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Ensure VADER is downloaded
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

def get_sentiment(text):
    scores = sia.polarity_scores(text)
    return scores

# Flask server
server = Flask(__name__)


@server.route('/api/sentiment', methods=['POST'])
def analyze_sentiment_api():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Invalid request. 'text' key is required."}), 400
    text_to_analyze = data['text']
    sentiment_scores = get_sentiment(text_to_analyze)
    return jsonify(sentiment_scores)

# Dash app
app = dash.Dash(__name__, server=server, url_base_pathname='/')

app.layout = html.Div(
    style={'fontFamily': 'Arial, sans-serif', 'maxWidth': '800px', 'margin': 'auto', 'padding': '20px',
           'backgroundColor': '#f9f9f9', 'borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'},
    children=[
        html.H1('Tweet Sentiment Classifier', style={'textAlign': 'center', 'color': '#333'}),
        html.P('Enter text below to classify its sentiment.', style={'textAlign': 'center', 'color': '#666'}),

        dcc.Textarea(
            id='text-input',
            placeholder='Enter your tweet here...',
            style={'width': '100%', 'height': 150, 'padding': '10px',
                   'borderRadius': '5px', 'border': '1px solid #ccc', 'fontSize': '16px'}
        ),

        html.Button(
            'Classify Sentiment', id='analyze-button', n_clicks=0,
            style={'display': 'block', 'margin': '20px auto', 'padding': '10px 20px',
                   'fontSize': '16px', 'cursor': 'pointer', 'backgroundColor': '#007BFF',
                   'color': 'white', 'border': 'none', 'borderRadius': '5px'}
        ),

        dcc.Loading(
            id="loading-spinner",
            type="circle",
            children=html.Div(id='sentiment-result')
        )
    ]
)

@app.callback(
    Output('sentiment-result', 'children'),
    [Input('analyze-button', 'n_clicks')],
    [State('text-input', 'value')]
)
def update_output(n_clicks, text_value):
    if n_clicks == 0 or not text_value:
        return html.P("Enter text and click classify.", style={'textAlign': 'center', 'color': '#888'})

    scores = get_sentiment(text_value)
    sentiment = "Positive" if scores['compound'] > 0.05 else "Negative" if scores['compound'] < -0.05 else "Neutral"

    return html.Div([
        html.H3(f"Predicted Sentiment: {sentiment}", style={'textAlign': 'center', 'color': '#007BFF'}),
        dcc.Graph(
            figure=go.Figure(
                data=[go.Bar(
                    x=['Negative', 'Neutral', 'Positive'],
                    y=[scores['neg'], scores['neu'], scores['pos']],
                    marker_color=['#FF6347', '#D3D3D3', '#32CD32']
                )]
            ).update_layout(
                title="Sentiment Breakdown",
                yaxis=dict(range=[0, 1]),
                plot_bgcolor='white',
                paper_bgcolor='#f9f9f9'
            )
        )
    ])

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8050))
    server.run(debug=False, port=port, host='0.0.0.0')
