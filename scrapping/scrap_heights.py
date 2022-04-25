import pandas as pd
import re
from selenium import webdriver
from bs4 import BeautifulSoup
from unidecode import unidecode


# Setting the driver

#Pour Firefox Linux
driver = webdriver.Firefox(executable_path="/home/elie/Documents/MoneyBallReloaded/gecko/geckodriver")

# Pour Google Chrome Windows. Il faut installer cependant chromedrive: https://sites.google.com/a/chromium.org/chromedriver/downloads
# driver = webdriver.Chrome(executable_path="C:\Program Files (x86)\Google\Chrome\Application\chromedriver_win32\chromedriver.exe")

# get the content of the html page
driver.get("https://basketballnoise.com/nba-players-height-2019-2020/")

heights = {}

content = driver.page_source
soup = BeautifulSoup(content,features="lxml")
div = soup.find("div", attrs={'class':'entry-content'})
name = ""
value = 0

for tbody in div.findAll('tbody'):
    for tr in tbody.findAll("tr"):
        tds = tr.findAll("td")
        name = tds[0].text.replace("\n", "")
        value= tds[1].text
                 
        '''if (len(tds) == 3):
                value = str(tds[0].text)'''
        heights[name] = value
        
clean_dict = {key: value for key, value in heights.items() if re.match('^\d[\',.,’]\d+$', value)}

#transform to df
df = pd.DataFrame(list(clean_dict.items()),columns = ['Name','Height (ft)'])
df['Height (ft)'] = df['Height (ft)'].str.replace('’','.')
df['Height (cm)'] = pd.to_numeric(df['Height (ft)'].str.split('.').str[0])*30.48+pd.to_numeric(df['Height (ft)'].str.split('.').str[1])*2.54
df['Height (cm)'] = df['Height (cm)'].apply(round, args=[2])

print("Script to gather players height in cm is done.")

path="/home/elie/Documents/MoneyBallReloaded/csv/players_height.csv"
df.to_csv(path)
