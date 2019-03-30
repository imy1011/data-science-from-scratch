'''
Created on Jul 7, 2017
By Loan Vo
Original File Path: /Users/loanvo/GitHub/py/joelgrus/joelgrus/src/c09_getting_data.py

Descriptions:
1. STDIN AND STDOUT 
2. READING FILES
3. DELIMITED FILES
4. SCRAPING THE WEB: HTML AND THE PARSING THEREOF
5. USING APIs
'''





'''
1. STDIN AND STDOUT 
'''
# In the command line (Terminal window) (not in python window), we can call python scripts with input 
# parameters accessed within the script by sys.argv and input stream from stdin

# the egrep.py to find lines (streaming from stdin) which have a matching pattern and then pipe those 
# lines into line_count.py to count how many of them
# cat /Users/loanvo/GitHub/py/joelgrus/joelgrus/data/SomeFile.txt | python /Users/loanvo/GitHub/py/joelgrus/joelgrus/src/egrep.py '[0-9]' | python /Users/loanvo/GitHub/py/joelgrus/joelgrus/src/line_count.py

# most_common_words.py:counts the words in its input and writes out the most common ones
# cat /Users/loanvo/GitHub/py/joelgrus/joelgrus/data/SomeFile.txt | python /Users/loanvo/GitHub/py/joelgrus/joelgrus/src/most_common_words.py 5

'''
2. READING FILES
To read/write from/to a file in Python:
1. open it for read/write mode
2. close it when you're done
--> use a WITH block to avoid forgetting closing it.
'''
import re
with open('/Users/loanvo/GitHub/py/joelgrus/joelgrus/data/reading_file.txt','r') as f:
    for line in f:
        if line.strip():
            print(line)
        if re.search('the',line.lower()):
            print("--> Found 'the'")
 
with open('/Users/loanvo/GitHub/py/joelgrus/joelgrus/data/SomeFile.txt','a') as f:  
    f.write('\nLoan added this line into an exisiting file by opening the file in appending mode!')
    
with open('/Users/loanvo/GitHub/py/joelgrus/joelgrus/data/writing_file.txt','w') as f:  
    f.write("\na. If the file hasn't existed yet, a new file will be created with these content.")
    f.write("\nb. If the file exists, it will be overwritten with these lines.")
    f.write("\nNotes: new file created and overwritten the exisiting one at the beginning.")
    f.write("\nIt means that all these lines, and not only this final line, appear in the final file.")
    
'''
3. DELIMITED FILES
Usually data file has lots of data on each line. These fields are usually separated either by comma or tab
Sometimes, they include mixture of field separation such as comma, tab, colon, etc.
To parse them:
    a. Use Pandas package.
    b. Use Python's csv module (in BINARY mode (i.e. rb/wb) for python2, but rt/wt (text mode) in python 3 
'''
import csv
with open('/Users/loanvo/GitHub/py/joelgrus/joelgrus/data/tab_delimited_stock_prices.txt','rt') as f:
    reader = csv.reader(f, delimiter='\t')
    print("***********")
    print(type(reader))
    print(reader)
    print("***********")
    for row in reader:
        print(row)
        date = row[0]
        stock_symbol = row[1]
        closing_price = float(row[2])
        print("On", date, "the stock", stock_symbol, "has a closing price of", closing_price)
        

with open('/Users/loanvo/GitHub/py/joelgrus/joelgrus/data/colon_delimited_stock_prices.txt','rt') as f:
    reader = csv.DictReader(f,delimiter=":")
    for row in reader:
        date = row["date"]
        symbol = row["symbol"]
        closing_price = row["closing_price"]
        print("On", date, "the stock", symbol, "has a closing_price of", closing_price)

with open('/Users/loanvo/GitHub/py/joelgrus/joelgrus/data/colon_delimited_stock_prices_without_header.txt','rt') as f:
    reader = csv.DictReader(f, fieldnames = ["ngay", "ten", "gia"], delimiter=":")
    for row in reader:
        date = row["ngay"]
        symbol = row["ten"]
        closing_price = row["gia"]
        print("On", date, "the stock", symbol, "has a closing_price of", closing_price)
    
    
# For string that have delimiter as a part of its content, csv.writer will automatically, put the string
# in "". In the following example, 'HaveComma,InName' will be put in "" as written into file.
today_prices = { 'AAPL' : 90.91, 'MSFT' : 41.68, 'HaveComma,InName': 3.2, 'FB' : 64.5 }
with open('/Users/loanvo/GitHub/py/joelgrus/joelgrus/data/comma_delimiter_csv_write.txt', 'wt') as f:
    f_writer = csv.writer(f, delimiter = ',')
    for stock, price in today_prices.items():
        f_writer.writerow([stock, price])



'''
4. SCRAPING THE WEB: HTML AND THE PARSING THEREOF
'''
from bs4 import BeautifulSoup
import requests

#url = "https://en.wikipedia.org/wiki/G20"
url = "https://www.safaribooksonline.com/search/?query=data"
soup = BeautifulSoup(requests.get(url).text, 'html5lib')
tds = soup('td', '')
print(tds)
       
'''        
5. USING APIs: Using Twython
'''
# Search API: https://dev.twitter.com/rest/reference/get/search/tweets
from twython import Twython
twitter = Twython("15oXsPfhGg1Hq7Ylb0vJTKPGQ", "FVjnkj3t0zQXrXvTRCfWBE0LNtUTPmHrS1pAEIIeBbbmH3gqfk")
# search for tweets contiang the phrase "data scinece"
for status in twitter.search(q='"data science"')["statuses"]:
    user = status["user"]["screen_name"].encode('utf-8')
    text = status["text"].encode('utf-8')
    print(user,":",text)

# Streaming API: http://bit.ly/1ycOEgG
from twython import TwythonStreamer # this is a class for Twitter streaming handling
tweets = [] # should avoid using a global variable like this. But for simplicity, we use it here
class MyTwythonStreamer(TwythonStreamer): 
    # TwythonStreamer class can be initialized with (self, app_key, app_secret, oauth_token, oauth_token_secret)
    # --> when initialize MyTwythonStreamer class object we also need to use: 
    # MyTwythonStreamer(app_key, app_secret, oauth_token, oauth_token_secret)
    # in TwythonStreamer class, there is two methods that we would like to modify:
    # 1. on_success(self, data)
    # 2. on_error(self, status_code, data)

    def on_success(self, data):
        if data['lang'] == 'en':
            tweets.append(data)
            print("Received tweet #", len(tweets))
        if len(tweets) >= 1000:
            self.disconnect()
    def on_error(self, status_code, data):
        print(status_code, data)
        self.disconnect()
        
# Now let pull a stream of data from Twitter
stream = MyTwythonStreamer("15oXsPfhGg1Hq7Ylb0vJTKPGQ", "FVjnkj3t0zQXrXvTRCfWBE0LNtUTPmHrS1pAEIIeBbbmH3gqfk", 
                           "884418469529362432-PbQhE1GyNVIEOqSuIZtKAx8delANdQA", 
                           "nJLtIDUFnDGTz0r0HIwGSjth0dqBcGMTGEuRkGiRA3Nr4")
print(stream.statuses.filter(track='data')) #consuming public statuses that contain the keyword 'data'
print("**********************************")
#print(stream.statuses.sample()) # consuming a sample of ALL public statues
print("**********************************")
from collections import Counter
top_hashtag = Counter(hashtag['text'].lower() for tweet in tweets for hashtag in tweet['entities']['hashtags'] )
print(top_hashtag.most_common(3))



