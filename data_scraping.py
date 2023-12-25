# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 22:35:37 2023

@author: hieu
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd 

# list of all pages that contain Nintendo Switch games
url_list = ['https://en.wikipedia.org/wiki/List_of_Nintendo_Switch_games_(0%E2%80%939_and_A)',
            'https://en.wikipedia.org/wiki/List_of_Nintendo_Switch_games_(B)',
            'https://en.wikipedia.org/wiki/List_of_Nintendo_Switch_games_(C%E2%80%93G)',
            'https://en.wikipedia.org/wiki/List_of_Nintendo_Switch_games_(H%E2%80%93P)',
            'https://en.wikipedia.org/wiki/List_of_Nintendo_Switch_games_(Q%E2%80%93Z)',
            ]
# Go through each page and get the data table from it. Count the total number of games
table_list = []
total_games = 0
for url in url_list:
    get_url = requests.get(url).text
    soup = BeautifulSoup(get_url, 'lxml')
    table = soup.find('table', class_='wikitable plainrowheaders sortable')
    
    num_games_in_table = len(table.find_all('tr'))
    total_games += num_games_in_table
    
    table_list.append(table)
    
# Initiate an empty list to hold the data
data = []
# Create two variable 'failed' and j to keep track the progress
failed = False
j = 0
# Iterate each table from the table list
for table in table_list:
    # Iterate each row in the table
    for row in table.find_all('tr'):         
        try:
            title = row.find_all('th')
            # Get the wiki link of the game
            link = title[0].find(href=True)['href']
            # Get the game title
            name = title[0].find(text=True)
            
        except:
            failed = True
            pass
        
        else:
            
            # Get other attributes: developer, publisher, and release date
            other_atts = row.find_all('td')
            att_list = []
            for i in range(len(other_atts) - 1):
                att = other_atts[i].find(text=True)
                att_list.append(att)
                
            # Get the gameplay, plot, genre, and game mode from each game
            
            wiki_game_url = 'https://en.wikipedia.org' + link  
            game_url = requests.get(wiki_game_url).text
            game_soup = BeautifulSoup(game_url, 'lxml')
            
            # Get mode and genre
            try:
                game_table = game_soup.find('table', class_='infobox ib-video-game hproduct')
                last_row = game_table.find_all('tr')[-1].text
                second_last_row = game_table.find_all('tr')[-2].text
                if last_row.startswith('Mode'):
                    mode = last_row
                    if second_last_row.startswith("Genre"):
                        genre = second_last_row
                    else:
                        genre = None
                elif last_row.startswith('Genre'):
                    mode = None
                    genre = last_row
                else:
                    mode = None
                    genre = None
            except:
                mode = None
                genre = None
                
            # Get gameplay and plot
            gameplay_text = ''
            plot_text = ''   
            for part in game_soup.find_all('h2'):               
                if part.text.startswith('Game'):
                    gameplay_text += part.text + '\n'
                    for element in part.next_siblings:
                        if element.name and element.name.startswith('h'):
                            break
                        elif element.name == 'p':
                            gameplay_text += element.text
                else:
                    pass
                                    
                if part.text.startswith('Plot'):
                    plot_text += part.text + '\n'
                    for element in part.next_siblings:
                        if element.name and element.name.startswith('h'):
                            break
                        elif element.name == 'p':
                            plot_text += element.text
                else:
                    pass                
            if gameplay_text == '':
                gameplay_text = None
            if plot_text == '':
                plot_text = None
            
            # Append all the collected data to the data list
            data.append([name, link] + att_list + [genre, mode, gameplay_text, plot_text])
            
        # Keep track the progress    
        j += 1
        if failed:
            print("Progess: {}/{}. Failed to get data.".format(j, total_games))
            failed = False
        else:
            print("Progess: {}/{}.".format(j, total_games))
        
print('DONE!!!!!')     
 
# Create a dataframe from the collected data
columns = ['Title', 'Url', 'Developers', 'Publisher', 'ReleaseDate', 'Genre', 'Mode', 'Gameplay', 'Plot']
df_games = pd.DataFrame(data=data, columns=columns)
# Save the data to csv file       
df_games.to_csv('Datasets/NintendoSwitchGames.csv', index=False)
