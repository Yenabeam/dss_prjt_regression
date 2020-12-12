#!/usr/bin/env python
# coding: utf-8

# In[6]:


from bs4 import BeautifulSoup
import urllib.parse as urlparse
import requests
import pandas as pd
import configparser


# In[41]:


url = "http://apis.data.go.kr/B090041/openapi/service/SpcdeInfoService"
operation = "getRestDeInfo"
config = configparser.ConfigParser()
config.read('/root/naver.ini')
mykey = key['public_api_key'].replace('_', '%')
date = []
datename = []
for year in range(2018, 2020):
    year = str(year)
    
    for month in range(1, 13):
        if month < 10:
            month = "0" + str(month)
        else:
            month = str(month)
            
        params = {'solYear' : year, 'solMonth' : month}
        rq_query = url +'/' + operation + '?' + urlparse.urlencode(params) + "&serviceKey=" + mykey    
        response = requests.get(rq_query) 
        dom = BeautifulSoup(response.content, "html.parser")
        
        items = dom.find_all("item")
        for item in items:
            date.append(item.locdate.string)
            datename.append(item.datename.string)
            
holiday_df= pd.DataFrame({"date": date, "datename": datename})
holiday_df['date'] = holiday_df['date'].astype('int')
holiday_df = holiday_df[holiday_df['date']>=20181001]
holiday_df = holiday_df[holiday_df['date']<=20190930]

holiday_df.to_csv('national_holiday.csv')



