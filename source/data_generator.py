from fycharts.SpotifyCharts import SpotifyCharts
from joblib import Parallel, delayed

def get_data(date_range):
    print("Dates are {} {}".format(date_range[0], date_range[1]))
    charts_api = SpotifyCharts()
    charts_api.top200Weekly("top200weekly"+date_range[0]+".csv", start=date_range[0], end=date_range[1], region=["us"])

date_ranges = [("2017-01-06", "2017-06-01"),( "2017-06-02", "2017-11-02"), ("2017-11-03", "2018-04-05"),
               ("2018-04-06", "2018-09-06"), ("2018-09-07", "2019-02-07"), ("2019-02-08", "2019-07-04"),
               ("2019-07-05", "2019-12-05"),("2019-12-06", "2020-03-26")]

Parallel(n_jobs=8,verbose=10)(delayed(get_data)(dates) for dates in date_ranges)

