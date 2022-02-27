# -*- coding: utf-8 -*-
"""
    Common utilities to be used in application
"""

import os

import datetime

from numpy import number

# Instance folder path, to keep stuff aware from flask app.
INSTANCE_FOLDER_PATH = os.path.join('/tmp', 'flaskstarter-instance')


# Form validation

NAME_LEN_MIN = 4
NAME_LEN_MAX = 25

PASSWORD_LEN_MIN = 6
PASSWORD_LEN_MAX = 16


# Model
STRING_LEN = 64

def get_current_time():
    return datetime.datetime.utcnow()


def pretty_date(dt, default=None):
    # Returns string representing "time since" eg 3 days ago, 5 hours ago etc.

    if default is None:
        default = 'just now'

    now = datetime.datetime.utcnow()
    diff = now - dt

    periods = (
        (diff.days / 365, 'year', 'years'),
        (diff.days / 30, 'month', 'months'),
        (diff.days / 7, 'week', 'weeks'),
        (diff.days, 'day', 'days'),
        (diff.seconds / 3600, 'hour', 'hours'),
        (diff.seconds / 60, 'minute', 'minutes'),
        (diff.seconds, 'second', 'seconds'),
    )

    for period, singular, plural in periods:

        if not period:
            continue

        if int(period) >= 1:
            if int(period) > 1:
                return u'%d %s ago' % (period, plural)
            return u'%d %s ago' % (period, singular)

    return default

import joblib
import pandas as pd
MODEL = joblib.load('/Users/athul/git/adStar/flaskstarter/DT_las_gbdt.pkl')

chunksize = 10 ** 6

filename = "~/Downloads/reportinghour.csv"
unique_screen = {}
line_count = 0
with pd.read_csv(filename, chunksize=chunksize) as reader:
  for chunk in reader:
    for index, row in chunk.iterrows():
        line_count = line_count+1
        unique_screen[row['ScreenId']] = row
    if line_count % 1000000== 0:
        print(line_count)

def get_impressions(data):
    # return ""
    model_input = adapt_input(data)
    return MODEL.predict(model_input)
    
def adapt_input(data):
    print(data)

    data['schedule'] = pd.to_datetime(data['schedule'], format='%a-%d')
    assert data['schedule'].isnull().sum() == 0, "missing date"
    data['Date_hour'] = data['schedule'].hour
    data['Date_dayofweek'] = data['schedule'].dayofweek
    number_of_screens = []
    cities = []
    venue_ids = []
    for screen in data['screens']:
        number_of_screens.append(unique_screen[screen]['NumberOfScreens'])
        cities.append(unique_screen[screen]['City'])
        venue_ids.append(unique_screen[screen]['VenueId'])

    data['VenueId'] = venue_ids
    data['City'] = cities
    data['NumberOfScreens'] = number_of_screens
    del data['schedule']
    df = pd.DataFrame.from_dict(data)
    return df