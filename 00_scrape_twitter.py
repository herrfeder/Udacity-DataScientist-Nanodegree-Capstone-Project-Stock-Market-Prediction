# 00 Scraping Tweets for Keyword Sentiments

import twint
from datetime import date, timedelta
import time

# I'm using this function with the library/tool twint (https://github.com/twintproject/twint) that allows for APIless Twitter Scraping to collect a maximum of 500 tweets for every day 
# between 00:00:00 and 15:00:00 for the last 5 years

def collect_daily_tweets(search):
    '''
    Will collect tweets using twint without the official Twitter API using the URI Query Interface.
    Function is hardcoded for collect 500 daily tweets for the last five years.
    
    INPUT:
        search - (str) Search String or Hashtag that has to be included in scraped tweets
        
    OUTPUT:
        None - Tweets will be saved into seperate folder with the pattern "{search_string}+{YYY-MM-DD00:00:00}.csv"
    
    '''
    start_date = date(2015, 1, 1)
    end_date = date(2020, 3, 10)
    delta = timedelta(days=1)
    while start_date <= end_date:
        start_string = start_date.strftime("%Y-%m-%d") + " 00:00:00"
        end_string = start_date.strftime("%Y-%m-%d") + " 15:00:00"
        c = twint.Config()
        c.Search = search
        c.Since = start_string
        c.Until = end_string
        c.Lang = "en"
        c.Limit = 500
        c.Store_csv = True
        c.Output = search + start_string.split(" ")[0]
        
        twint.run.Search(c)
        start_date += delta
        time.sleep(30)
        
if __name__ == "__main__":

    collect_daily_tweets("#economy")