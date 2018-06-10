

```python
import tweepy
import json
import pandas as pd
import numpy as np
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from config import (consumer_key, consumer_secret, 
                    access_token, access_token_secret)
```


```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
```


```python
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())
```


```python
#Get tweets from each news organization's twitter feeds.



# A list to hold sentiments.

sentiments = []

target_users = ("@BBC", "@CBS", "@CNN", "@FoxNews", "@NYTimes")

#Loop though target users.

for user in target_users:
    
    #Counter 

    counter = 0
    
    #Loop through 5 pages of tweets for each news organization.


    #Get all tweets from the homefeed of each news organization.

    public_tweets = api.user_timeline(user, count = 100)

        #Loop through all tweets.

    for tweet in public_tweets:

        #Run the Vader analysis on each tweet.

        compound = analyzer.polarity_scores(tweet["text"])["compound"]
        pos = analyzer.polarity_scores(tweet["text"])["pos"]
        neu = analyzer.polarity_scores(tweet["text"])["neu"]
        neg = analyzer.polarity_scores(tweet["text"])["neg"]
        tweets_ago = counter
        tweet_text = tweet["text"]

        #Add sentiments for each tweet to the sentiments list.

        sentiments.append({"User" : user,
                           "Date": tweet["created_at"],
                           "Compound" : compound,
                           "Positive" : pos,
                           "Negative" : neg,
                           "Neutral" : neu,
                           "Tweets Ago" : counter,
                           "Tweet Text" : tweet_text})
        #Add to counter.

        counter = counter + 1
```


```python
news_sentiments = pd.DataFrame.from_dict(sentiments)
news_sentiments
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Compound</th>
      <th>Date</th>
      <th>Negative</th>
      <th>Neutral</th>
      <th>Positive</th>
      <th>Tweet Text</th>
      <th>Tweets Ago</th>
      <th>User</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.3612</td>
      <td>Thu Jun 07 19:04:09 +0000 2018</td>
      <td>0.143</td>
      <td>0.857</td>
      <td>0.000</td>
      <td>#OurGirl's Michelle Keegan discovers her great...</td>
      <td>0</td>
      <td>@BBC</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0000</td>
      <td>Thu Jun 07 18:00:21 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>❤️️ Harry has been wearing a Spider-Man mask d...</td>
      <td>1</td>
      <td>@BBC</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.7964</td>
      <td>Thu Jun 07 16:02:03 +0000 2018</td>
      <td>0.000</td>
      <td>0.530</td>
      <td>0.470</td>
      <td>Turns out even Shakespearean acting legends lo...</td>
      <td>2</td>
      <td>@BBC</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.4404</td>
      <td>Thu Jun 07 15:36:48 +0000 2018</td>
      <td>0.000</td>
      <td>0.884</td>
      <td>0.116</td>
      <td>RT @BBC6Music: 💪 Who do you hope to hear durin...</td>
      <td>3</td>
      <td>@BBC</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.8043</td>
      <td>Thu Jun 07 15:32:24 +0000 2018</td>
      <td>0.086</td>
      <td>0.544</td>
      <td>0.369</td>
      <td>This elderly man was struggling slowly across ...</td>
      <td>4</td>
      <td>@BBC</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-0.7906</td>
      <td>Thu Jun 07 14:04:02 +0000 2018</td>
      <td>0.269</td>
      <td>0.731</td>
      <td>0.000</td>
      <td>🐍 A Texan required 26 doses of anti-venom afte...</td>
      <td>5</td>
      <td>@BBC</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-0.4588</td>
      <td>Thu Jun 07 13:00:43 +0000 2018</td>
      <td>0.138</td>
      <td>0.862</td>
      <td>0.000</td>
      <td>RT @BBC_ARoadshow: 'We did it for all women' -...</td>
      <td>6</td>
      <td>@BBC</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.2263</td>
      <td>Thu Jun 07 13:00:26 +0000 2018</td>
      <td>0.000</td>
      <td>0.921</td>
      <td>0.079</td>
      <td>🎞🎬 From the Jurassic World sequel to an Oscar ...</td>
      <td>7</td>
      <td>@BBC</td>
    </tr>
    <tr>
      <th>8</th>
      <td>-0.5574</td>
      <td>Thu Jun 07 12:01:05 +0000 2018</td>
      <td>0.242</td>
      <td>0.659</td>
      <td>0.099</td>
      <td>👊💕 If you've ever been touched by cancer, Juli...</td>
      <td>8</td>
      <td>@BBC</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.6531</td>
      <td>Thu Jun 07 11:46:22 +0000 2018</td>
      <td>0.057</td>
      <td>0.706</td>
      <td>0.237</td>
      <td>RT @BBCEarth: Are you obsessed with the ocean?...</td>
      <td>9</td>
      <td>@BBC</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.7003</td>
      <td>Thu Jun 07 11:01:05 +0000 2018</td>
      <td>0.000</td>
      <td>0.784</td>
      <td>0.216</td>
      <td>💕🐮 Cow cuddling is the new wellness trend comi...</td>
      <td>10</td>
      <td>@BBC</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.0000</td>
      <td>Thu Jun 07 09:37:40 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>Little girls don’t stay little forever. https:...</td>
      <td>11</td>
      <td>@BBC</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.3612</td>
      <td>Thu Jun 07 08:01:03 +0000 2018</td>
      <td>0.000</td>
      <td>0.828</td>
      <td>0.172</td>
      <td>What it's like to find out you were adopted at...</td>
      <td>12</td>
      <td>@BBC</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.3182</td>
      <td>Thu Jun 07 07:28:01 +0000 2018</td>
      <td>0.000</td>
      <td>0.892</td>
      <td>0.108</td>
      <td>The original illustrated map of The Hundred Ac...</td>
      <td>13</td>
      <td>@BBC</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.5106</td>
      <td>Thu Jun 07 07:01:05 +0000 2018</td>
      <td>0.000</td>
      <td>0.732</td>
      <td>0.268</td>
      <td>❤️🐶 Some heroes don't wear capes - they wear c...</td>
      <td>14</td>
      <td>@BBC</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.0000</td>
      <td>Wed Jun 06 20:44:01 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>Tonight, award-winning writer Jeanette Winters...</td>
      <td>15</td>
      <td>@BBC</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.3818</td>
      <td>Wed Jun 06 20:04:00 +0000 2018</td>
      <td>0.000</td>
      <td>0.860</td>
      <td>0.140</td>
      <td>Comedian @JordBrookes joins @SusanCalman for #...</td>
      <td>16</td>
      <td>@BBC</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.0000</td>
      <td>Wed Jun 06 19:03:03 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>Tonight, #OurGirl actress @michkeegan uncovers...</td>
      <td>17</td>
      <td>@BBC</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.1779</td>
      <td>Wed Jun 06 18:03:04 +0000 2018</td>
      <td>0.122</td>
      <td>0.718</td>
      <td>0.160</td>
      <td>No leaks (and better orgasms) - what you need ...</td>
      <td>18</td>
      <td>@BBC</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.0000</td>
      <td>Wed Jun 06 17:02:09 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>74 years on from D-Day, these infrared photos ...</td>
      <td>19</td>
      <td>@BBC</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.0000</td>
      <td>Wed Jun 06 16:00:01 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>There are more than 110,000 gang members in Ho...</td>
      <td>20</td>
      <td>@BBC</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.3182</td>
      <td>Wed Jun 06 13:02:05 +0000 2018</td>
      <td>0.000</td>
      <td>0.892</td>
      <td>0.108</td>
      <td>From a French novel about the Burundi genocide...</td>
      <td>21</td>
      <td>@BBC</td>
    </tr>
    <tr>
      <th>22</th>
      <td>-0.5574</td>
      <td>Wed Jun 06 12:21:20 +0000 2018</td>
      <td>0.153</td>
      <td>0.847</td>
      <td>0.000</td>
      <td>RT @BBC_ARoadshow: Fiona discovers more about ...</td>
      <td>22</td>
      <td>@BBC</td>
    </tr>
    <tr>
      <th>23</th>
      <td>-0.6523</td>
      <td>Wed Jun 06 12:03:06 +0000 2018</td>
      <td>0.236</td>
      <td>0.764</td>
      <td>0.000</td>
      <td>Ethan Hawke relives the time when one of his c...</td>
      <td>23</td>
      <td>@BBC</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0.0000</td>
      <td>Wed Jun 06 11:02:02 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>In 1993, Steven Spielberg's film Jurassic Park...</td>
      <td>24</td>
      <td>@BBC</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0.0000</td>
      <td>Wed Jun 06 10:01:40 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>RT @BBCSpringwatch: Ever heard a beatboxing st...</td>
      <td>25</td>
      <td>@BBC</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0.4215</td>
      <td>Wed Jun 06 08:55:39 +0000 2018</td>
      <td>0.000</td>
      <td>0.833</td>
      <td>0.167</td>
      <td>RT @bbcthree: Madame Poole has been ballet dan...</td>
      <td>26</td>
      <td>@BBC</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0.0000</td>
      <td>Wed Jun 06 08:55:13 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>RT @BBCTwo: What if @BBCSpringwatch did #LoveI...</td>
      <td>27</td>
      <td>@BBC</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0.0000</td>
      <td>Wed Jun 06 08:55:07 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>RT @BBCFOUR: Here's a sneak preview of our bra...</td>
      <td>28</td>
      <td>@BBC</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0.2732</td>
      <td>Wed Jun 06 08:02:01 +0000 2018</td>
      <td>0.126</td>
      <td>0.698</td>
      <td>0.177</td>
      <td>When Mona began to lose her sight, her 84-year...</td>
      <td>29</td>
      <td>@BBC</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>470</th>
      <td>0.0000</td>
      <td>Thu Jun 07 13:00:18 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>Morning Briefing: Here's what you need to know...</td>
      <td>70</td>
      <td>@NYTimes</td>
    </tr>
    <tr>
      <th>471</th>
      <td>-0.4019</td>
      <td>Thu Jun 07 12:46:06 +0000 2018</td>
      <td>0.144</td>
      <td>0.856</td>
      <td>0.000</td>
      <td>He delivered pizza to an Army base in Brooklyn...</td>
      <td>71</td>
      <td>@NYTimes</td>
    </tr>
    <tr>
      <th>472</th>
      <td>0.0000</td>
      <td>Thu Jun 07 12:30:08 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>Afghanistan's president declared a brief unila...</td>
      <td>72</td>
      <td>@NYTimes</td>
    </tr>
    <tr>
      <th>473</th>
      <td>-0.8126</td>
      <td>Thu Jun 07 12:22:05 +0000 2018</td>
      <td>0.296</td>
      <td>0.704</td>
      <td>0.000</td>
      <td>No, Canada did not burn down the White House d...</td>
      <td>73</td>
      <td>@NYTimes</td>
    </tr>
    <tr>
      <th>474</th>
      <td>0.6597</td>
      <td>Thu Jun 07 12:10:08 +0000 2018</td>
      <td>0.000</td>
      <td>0.795</td>
      <td>0.205</td>
      <td>Amazon won the right to show Premier League ga...</td>
      <td>74</td>
      <td>@NYTimes</td>
    </tr>
    <tr>
      <th>475</th>
      <td>-0.4404</td>
      <td>Thu Jun 07 12:00:01 +0000 2018</td>
      <td>0.196</td>
      <td>0.804</td>
      <td>0.000</td>
      <td>Your daily @DealBook Briefing:\n\n• Athenaheal...</td>
      <td>75</td>
      <td>@NYTimes</td>
    </tr>
    <tr>
      <th>476</th>
      <td>-0.6369</td>
      <td>Thu Jun 07 11:56:51 +0000 2018</td>
      <td>0.279</td>
      <td>0.721</td>
      <td>0.000</td>
      <td>Miss America says it will no longer judge cont...</td>
      <td>76</td>
      <td>@NYTimes</td>
    </tr>
    <tr>
      <th>477</th>
      <td>0.0000</td>
      <td>Thu Jun 07 11:50:00 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>Our analysis shows that Stephon Clark was shot...</td>
      <td>77</td>
      <td>@NYTimes</td>
    </tr>
    <tr>
      <th>478</th>
      <td>0.0000</td>
      <td>Thu Jun 07 11:47:33 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>We dissected the extensive video footage, stud...</td>
      <td>78</td>
      <td>@NYTimes</td>
    </tr>
    <tr>
      <th>479</th>
      <td>-0.6705</td>
      <td>Thu Jun 07 11:40:03 +0000 2018</td>
      <td>0.191</td>
      <td>0.809</td>
      <td>0.000</td>
      <td>In March, Stephon Clark encountered two police...</td>
      <td>79</td>
      <td>@NYTimes</td>
    </tr>
    <tr>
      <th>480</th>
      <td>-0.6249</td>
      <td>Thu Jun 07 11:31:04 +0000 2018</td>
      <td>0.242</td>
      <td>0.758</td>
      <td>0.000</td>
      <td>For months, American officials have been worri...</td>
      <td>80</td>
      <td>@NYTimes</td>
    </tr>
    <tr>
      <th>481</th>
      <td>0.0000</td>
      <td>Thu Jun 07 11:16:01 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>Police officers in Baltimore used brass knuckl...</td>
      <td>81</td>
      <td>@NYTimes</td>
    </tr>
    <tr>
      <th>482</th>
      <td>0.0000</td>
      <td>Thu Jun 07 11:00:03 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>North Korea demolished a "key missile test sta...</td>
      <td>82</td>
      <td>@NYTimes</td>
    </tr>
    <tr>
      <th>483</th>
      <td>0.1531</td>
      <td>Thu Jun 07 10:45:05 +0000 2018</td>
      <td>0.096</td>
      <td>0.780</td>
      <td>0.124</td>
      <td>Paul Ryan dismissed President Trump’s charge o...</td>
      <td>83</td>
      <td>@NYTimes</td>
    </tr>
    <tr>
      <th>484</th>
      <td>-0.1779</td>
      <td>Thu Jun 07 10:31:05 +0000 2018</td>
      <td>0.144</td>
      <td>0.856</td>
      <td>0.000</td>
      <td>After apologizing for using an epithet to desc...</td>
      <td>84</td>
      <td>@NYTimes</td>
    </tr>
    <tr>
      <th>485</th>
      <td>-0.3612</td>
      <td>Thu Jun 07 10:16:07 +0000 2018</td>
      <td>0.122</td>
      <td>0.878</td>
      <td>0.000</td>
      <td>The chief spokesman of Pakistan's army recentl...</td>
      <td>85</td>
      <td>@NYTimes</td>
    </tr>
    <tr>
      <th>486</th>
      <td>0.0000</td>
      <td>Thu Jun 07 10:05:03 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>Morning Briefing: Here's what you need to know...</td>
      <td>86</td>
      <td>@NYTimes</td>
    </tr>
    <tr>
      <th>487</th>
      <td>0.2732</td>
      <td>Thu Jun 07 09:46:04 +0000 2018</td>
      <td>0.106</td>
      <td>0.737</td>
      <td>0.157</td>
      <td>A commission that has been working for 4 years...</td>
      <td>87</td>
      <td>@NYTimes</td>
    </tr>
    <tr>
      <th>488</th>
      <td>-0.7506</td>
      <td>Thu Jun 07 09:31:04 +0000 2018</td>
      <td>0.316</td>
      <td>0.684</td>
      <td>0.000</td>
      <td>France's Parliament is set to begin debating a...</td>
      <td>88</td>
      <td>@NYTimes</td>
    </tr>
    <tr>
      <th>489</th>
      <td>0.0000</td>
      <td>Thu Jun 07 09:15:04 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>RT @jakesilverstein: This year’s New York Issu...</td>
      <td>89</td>
      <td>@NYTimes</td>
    </tr>
    <tr>
      <th>490</th>
      <td>0.0000</td>
      <td>Thu Jun 07 09:00:06 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>Many survivors of Ireland’s Magdalene Laundrie...</td>
      <td>90</td>
      <td>@NYTimes</td>
    </tr>
    <tr>
      <th>491</th>
      <td>-0.4404</td>
      <td>Thu Jun 07 08:44:06 +0000 2018</td>
      <td>0.146</td>
      <td>0.854</td>
      <td>0.000</td>
      <td>RT @nytimesworld: 220 survivors of Ireland’s n...</td>
      <td>91</td>
      <td>@NYTimes</td>
    </tr>
    <tr>
      <th>492</th>
      <td>-0.8316</td>
      <td>Thu Jun 07 08:30:12 +0000 2018</td>
      <td>0.350</td>
      <td>0.556</td>
      <td>0.095</td>
      <td>The husband of the influential designer Kate S...</td>
      <td>92</td>
      <td>@NYTimes</td>
    </tr>
    <tr>
      <th>493</th>
      <td>-0.8658</td>
      <td>Thu Jun 07 08:14:05 +0000 2018</td>
      <td>0.338</td>
      <td>0.662</td>
      <td>0.000</td>
      <td>RT @nytimesworld: Kamel Matmati, 21, was taken...</td>
      <td>93</td>
      <td>@NYTimes</td>
    </tr>
    <tr>
      <th>494</th>
      <td>-0.8807</td>
      <td>Thu Jun 07 08:02:06 +0000 2018</td>
      <td>0.427</td>
      <td>0.573</td>
      <td>0.000</td>
      <td>More State Dept. employees have fallen ill aft...</td>
      <td>94</td>
      <td>@NYTimes</td>
    </tr>
    <tr>
      <th>495</th>
      <td>0.0000</td>
      <td>Thu Jun 07 07:47:04 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>In the last few months, Kim Jong-un has engine...</td>
      <td>95</td>
      <td>@NYTimes</td>
    </tr>
    <tr>
      <th>496</th>
      <td>0.0000</td>
      <td>Thu Jun 07 07:32:06 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>The yacht at the center of Britain's costliest...</td>
      <td>96</td>
      <td>@NYTimes</td>
    </tr>
    <tr>
      <th>497</th>
      <td>0.0000</td>
      <td>Thu Jun 07 07:17:05 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>RT @nytimesworld: An Afghan police officer, am...</td>
      <td>97</td>
      <td>@NYTimes</td>
    </tr>
    <tr>
      <th>498</th>
      <td>0.3612</td>
      <td>Thu Jun 07 07:02:05 +0000 2018</td>
      <td>0.000</td>
      <td>0.902</td>
      <td>0.098</td>
      <td>"I've made this dish twice now and, just like ...</td>
      <td>98</td>
      <td>@NYTimes</td>
    </tr>
    <tr>
      <th>499</th>
      <td>0.0000</td>
      <td>Thu Jun 07 06:32:01 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>Greece has declared an economic recovery. Try ...</td>
      <td>99</td>
      <td>@NYTimes</td>
    </tr>
  </tbody>
</table>
<p>500 rows × 8 columns</p>
</div>




```python
news_sentiments.to_csv("Twitter_News_Mood.csv", index=False)
```


```python
plt.xlim(101, -1)

#plot scatterplot using a for loop.
for user in target_users:
    dataframe = news_sentiments.loc[news_sentiments["User"] == user]
    plt.scatter(dataframe["Tweets Ago"],dataframe["Compound"],label = user)
    
#Add legend
plt.legend(bbox_to_anchor = (1,1))

#Add title, x axis label, and y axis label.
plt.title("Sentiment Analysis of Media Tweets (11/5/2017)")
plt.xlabel("Tweets Ago")
plt.ylabel("Tweet Polarity")

#Set a grid on the plot.
plt.grid()

plt.savefig("Sentiment Analysis of Media Tweets")
plt.show()
```


![png](output_6_0.png)



```python
average_sentiment = news_sentiments.groupby("User")["Compound"].mean()
average_sentiment
```




    User
    @BBC        0.099469
    @CBS        0.314271
    @CNN        0.001074
    @FoxNews    0.060668
    @NYTimes   -0.132334
    Name: Compound, dtype: float64




```python
x_axis = np.arange(len(average_sentiment))
xlabels = average_sentiment.index
count = 0
for sentiment in average_sentiment:
    plt.text(count, sentiment+.01, str(round(sentiment,2)))
    count = count + 1
plt.bar(x_axis, average_sentiment, tick_label = xlabels, color = ['silver', 'b', 'y', 'g', 'c'])
#Set title, x axis label, and y axis label.
plt.title("Overall Sentiment of Media Tweets (11/5/2017)")
plt.xlabel("New Organizations")
plt.ylabel("Tweet Polarity")
plt.savefig("Overall Sentiment of Media Tweets")
plt.show()
```


![png](output_8_0.png)

