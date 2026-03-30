'''
PART 2: Pre-processing
- Take the time to understand the data before proceeding
- Load `pred_universe_raw.csv` into a dataframe and `arrest_events_raw.csv` into a dataframe
- Perform a full outer join/merge on 'person_id' into a new dataframe called `df_arrests`
- Create a column in `df_arrests` called `y` which equals 1 if the person was arrested for a felony crime in the 365 days after their arrest date in `df_arrests`. 
- - So if a person was arrested on 2016-09-11, you would check to see if there was a felony arrest for that person between 2016-09-12 and 2017-09-11.
- - Use a print statment to print this question and its answer: What share of arrestees in the `df_arrests` table were rearrested for a felony crime in the next year?
- Create a predictive feature for `df_arrests` that is called `current_charge_felony` which will equal one if the current arrest was for a felony charge, and 0 otherwise. 
- - Use a print statment to print this question and its answer: What share of current charges are felonies?
- Create a predictive feature for `df_arrests` that is called `num_fel_arrests_last_year` which is the total number arrests in the one year prior to the current charge. 
- - So if someone was arrested on 2016-09-11, then you would check to see if there was a felony arrest for that person between 2015-09-11 and 2016-09-10.
- - Use a print statment to print this question and its answer: What is the average number of felony arrests in the last year?
- Print the mean of 'num_fel_arrests_last_year' -> pred_universe['num_fel_arrests_last_year'].mean()
- Print pred_universe.head()
- Return `df_arrests` for use in main.py for PART 3; if you can't figure this out, save as a .csv in `data/` and read into PART 3 in main.py
'''

# import the necessary packages
import pandas as pd


# Your code here
def run_preprocessing():

    pred = pd.read_csv("data/pred_universe_raw.csv")
    events = pd.read_csv("data/arrest_events_raw.csv")

    pred['arrest_date_univ'] = pd.to_datetime(pred['arrest_date_univ'])
    events['arrest_date_event'] = pd.to_datetime(events['arrest_date_event'])

    df = pred.merge(events, on="person_id", how="outer")

    
    df['y'] = 0

    for i, row in df.iterrows():
        if pd.notnull(row['arrest_date_univ']):
            start = row['arrest_date_univ'] + pd.Timedelta(days=1)
            end = row['arrest_date_univ'] + pd.Timedelta(days=365)

            future_events = events[
                (events['person_id'] == row['person_id']) &
                (events['arrest_date_event'] >= start) &
                (events['arrest_date_event'] <= end) &
                (events['charge_type'] == 'F')
            ]

            if len(future_events) > 0:
                df.at[i, 'y'] = 1

    print("Share rearrested for felony:", df['y'].mean())

   
    df['current_charge_felony'] = (df['charge_type'] == 'F').astype(int)
    print("Share current felonies:", df['current_charge_felony'].mean())

    
    df['num_fel_arrests_last_year'] = 0

    for i, row in df.iterrows():
        if pd.notnull(row['arrest_date_univ']):
            start = row['arrest_date_univ'] - pd.Timedelta(days=365)
            end = row['arrest_date_univ'] - pd.Timedelta(days=1)

            past_events = events[
                (events['person_id'] == row['person_id']) &
                (events['arrest_date_event'] >= start) &
                (events['arrest_date_event'] <= end) &
                (events['charge_type'] == 'F')
            ]

            df.at[i, 'num_fel_arrests_last_year'] = len(past_events)

    print("Avg felony arrests last year:", df['num_fel_arrests_last_year'].mean())
    print(df.head())

    df.to_csv("data/df_arrests.csv", index=False)

    return df


