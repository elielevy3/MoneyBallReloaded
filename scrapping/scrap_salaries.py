#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 18:49:01 2021

@author: elie
"""
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# Setting the driver Pour Firefox Linux
# driver = webdriver.Firefox(executable_path="/home/elie/Documents/MoneyBallReloaded/gecko/geckodriver")

option = webdriver.ChromeOptions()
option.add_argument('headless')
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=option)

# Pour Google Chrome Windows. Il faut installer chromedrive: https://sites.google.com/a/chromium.org/chromedriver/downloads
# driver = webdriver.Chrome(executable_path="C:\Program Files (x86)\Google\Chrome\Application\chromedriver_win32\chromedriver.exe")

# get the content of the html page
driver.get("https://www.basketball-reference.com/contracts/players.html")

# we also need to get the current salaries in their contract
content = driver.page_source
soup = BeautifulSoup(content, features="lxml")
content = soup.find(id="player-contracts").findAll("tbody")[0]
salaries = {}

for player in content.findAll("tr"):
    tds = player.findAll("td")
    if (len(tds) > 0):
        name = tds[0].text
        salary = tds[2].text
        if (salary != ""):
            # salary = salary.replace("$", "").replace(",", "")
            salaries[name] = salary
    
# transform to df and write into csv
df = pd.DataFrame(list(salaries.items()),columns = ['Name','Salary'])
print("Script to gather players salaries is done.")
path="/home/elie/Documents/MoneyBallReloaded/csv/players_salaries.csv"
df.to_csv(path)
