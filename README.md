# Covid19-Dashboard

Data from the New York Times https://github.com/nytimes/covid-19-data

Dashboard shows visualizations of Covid19 cases and deaths in the US at the state and county level.

To change from state to country level, change 

`covidus.get_cases_dict(loc='state')`
`covidus.get_deaths_dict(loc='state')` 
`covidus.get_full_cases_dict(loc='state')`

to 

`covidus.get_cases_dict(loc='county')`
`covidus.get_deaths_dict(loc='county')` 
`covidus.get_full_cases_dict(loc='county')`

Set the starting data for all visualizations by using `covidus.set_start_date(start_date='yyyy-mm-dd')`

Set the number of states to show at once using `covidus.set_locations_to_show(num_locations)`

