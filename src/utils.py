import os
import requests
from .custom_logger import logger

def download_data(save_path='data/data.csv'):
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=1320&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=IPG2211A2N&scale=left&cosd=1939-01-01&coed=2024-10-01&line_color=%234572a7&link_values=false&line_style=solid&mark_type=none&mw=3&lw=3&ost=-99999&oet=99999&mma=0&fml=a&fq=Monthly&fam=avg&fgst=lin&fgsnd=2020-02-01&line_index=1&transformation=lin&vintage_date=2024-11-21&revision_date=2024-11-21&nd=1939-01-01"
    response = requests.get(url)
    response.raise_for_status()  # Check if the request was successful

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as file:
        file.write(response.content)
    
    logger.info(f"Data downloaded and saved to {save_path}")



