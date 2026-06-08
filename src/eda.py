import matplotlib.pyplot as plt
import pandas as pd
from streamlit import columns
import data_collection as dc
import config as cfg
import seaborn as sns   

# Define seasons
def get_season(month):
    if month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Fall'
    else:
        return 'Winter'
###############################################################################
# Generate resampled demand features.
# df: DataFrame with columns 'Local time' and 'Demand'
# resample_freq:  'D', 'W', 'ME' and 'YE'
# aggregation_func: 'sum', 'mean', 'median', 'min' and 'max', 'std'
# rolling_window: int, number of periods for rolling mean
# start_date: 'YYYY-MM-DD'
# end_date:  'YYYY-MM-DD'
#############################################################################
def generate_resampled_demand_features(
    df, resample_freq='D', aggregation_func='mean', rolling_window=10,
    start_date=None,
    end_date=None
    ):
    df = df.copy()
    # Remove commas and convert to numeric
    df['Demand'] = pd.to_numeric(
        df['Demand'].astype(str).str.replace(',', '', regex=False)
    )
    # Filter by date range if specified
    if start_date:
        df = df[df['Local time'] >= start_date]
    if end_date:
        df = df[df['Local time'] <= end_date]
    # Resample to resample_freq total demand
    Sample_freq_df = (
         df.set_index('Local time')
         [['Demand']]
         .resample(resample_freq)
         .agg(aggregation_func)
         .reset_index()
        )
    #Rolling mean
    Sample_freq_df[f'Rolling_{rolling_window}d'] = Sample_freq_df['Demand'].rolling(window=rolling_window).mean()   
    return Sample_freq_df

###############################################################################
# Plot electricity demand over time with rolling mean.
# df: DataFrame with columns 'Local time' and 'Demand'
# resample_freq:  'D', 'W', 'ME' and 'YE'
# aggregation_func: 'sum', 'mean', 'median', 'min' and 'max', 'std'
# rolling_window: int, number of periods for rolling mean
# start_date: 'YYYY-MM-DD'
# end_date:  'YYYY-MM-DD'
# csv_path: str or Path, optional, path to save the resampled data as CSV
# fig_path: str or Path, optional, path to save the plot as PNG or SVG
# xlabel: str, label for x-axis
# ylabel: str, label for y-axis
# title: str, title of the plot
#############################################################################
def plot_and_save( 
        Sample_freq_df, 
        resample_freq,
        rolling_window, 
        csv_path, fig_path,
        xlabel,
        ylabel,
        title
        ):
    
    plt.figure(figsize=(13,5))
    plt.plot(Sample_freq_df['Local time'], Sample_freq_df['Demand'], label=f'{title}', color='blue')
    plt.plot(Sample_freq_df['Local time'], Sample_freq_df[f'Rolling_{rolling_window}d'], label=f'Rolling Mean ({rolling_window} periods ({resample_freq}))', color='orange')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
     # Save to CSV and svg
    if csv_path:
        Sample_freq_df.to_csv(csv_path, index=False)
    else:
        Sample_freq_df.to_csv(cfg.REPORT_DIR / "electricity_demand.csv", index=False)

    if fig_path:
        plt.savefig(fig_path)
    else:
        plt.savefig(cfg.REPORT_DIR / "electricity_demand.svg", format='svg')

    plt.show()

######################################################################
# Plot daily mean electricity demand over time with rolling mean.
# df: DataFrame with columns 'Local time' and 'Demand'
# resample_freq:  'D', 'W', 'ME' and 'YE'
# aggregation_func: 'sum', 'mean', 'median', 'min' and 'max', 'std'
# start_date: 'YYYY-MM-DD'
# end_date:  'YYYY-MM-DD'
# rolling_window: int, number of periods for rolling mean
# csv_path: str or Path, optional, path to save the resampled data as CSV
# fig_path: str or Path, optional, path to save the plot as PNG or SVG
# xlabel: str, label for x-axis
# ylabel: str, label for y-axis
# title: str, title of the plot
#####################################################################
def plot_daily_mean_demand(
    df, resample_freq='D', aggregation_func='mean', rolling_window=15,
    start_date="2015-07-01",
    end_date="2026-05-30",
    csv_path=None, fig_path=None,
    xlabel="Date",
    ylabel="Mean Demand (MW)",
    title="Daily Mean Electricity Demand"
    ):
    # Generate resampled demand features
    Sample_freq_df= Sample_freq_df = generate_resampled_demand_features(
        df, resample_freq=resample_freq, aggregation_func=aggregation_func,
        rolling_window=rolling_window, start_date=start_date, end_date=end_date
    )

    # Plot and save 
    plot_and_save( 
        Sample_freq_df,
        resample_freq, 
        rolling_window, 
        csv_path, fig_path,
        xlabel,
        ylabel,
        title
        )
    return Sample_freq_df

######################################################################
# Plot monthly mean electricity demand over time with rolling mean.
# df: DataFrame with columns 'Local time' and 'Demand'
# resample_freq:  'D', 'W', 'ME' and 'YE'
# aggregation_func: 'sum', 'mean', 'median', 'min' and 'max', 'std'
# start_date: 'YYYY-MM-DD'
# end_date:  'YYYY-MM-DD'
# rolling_window: int, number of periods for rolling mean
# csv_path: str or Path, optional, path to save the resampled data as CSV
# fig_path: str or Path, optional, path to save the plot as PNG or SVG
# xlabel: str, label for x-axis
# ylabel: str, label for y-axis
# title: str, title of the plot
#####################################################################
def plot_monthly_mean_demand(
    df, resample_freq='M', aggregation_func='mean', rolling_window=5,
    start_date="2015-07-01",
    end_date="2026-05-30",
    csv_path= cfg.REPORT_DIR / "monthly_mean_demand.csv", 
    fig_path= cfg.REPORT_DIR / "monthly_mean_demand.png",
    xlabel="Month",
    ylabel="Mean Demand (MW)",
    
    title="Monthly Mean Electricity Demand"
    ):
    # Generate resampled demand features
    Sample_freq_df= Sample_freq_df = generate_resampled_demand_features(
        df, resample_freq=resample_freq, aggregation_func=aggregation_func,
        rolling_window=rolling_window, start_date=start_date, end_date=end_date
    )

    # Plot and save 
    plot_and_save( 
        Sample_freq_df,
        resample_freq, 
        rolling_window, 
        csv_path, fig_path,
        xlabel,
        ylabel,
        title
        )
    return Sample_freq_df

######################################################################
# Plot yearly mean electricity demand over time with rolling mean.
# df: DataFrame with columns 'Local time' and 'Demand'
# resample_freq:  'D', 'W', 'ME' and 'YE'
# aggregation_func: 'sum', 'mean', 'median', 'min' and 'max', 'std'
# start_date: 'YYYY-MM-DD'
# end_date:  'YYYY-MM-DD'
# rolling_window: int, number of periods for rolling mean
# csv_path: str or Path, optional, path to save the resampled data as CSV
# fig_path: str or Path, optional, path to save the plot as PNG or SVG
# xlabel: str, label for x-axis
# ylabel: str, label for y-axis
# title: str, title of the plot
#####################################################################

def plot_yearly_mean_demand(
    df, resample_freq='YE', aggregation_func='mean', rolling_window=2,
    start_date="2015-07-01",
    end_date="2026-05-30",
    csv_path= cfg.REPORT_DIR / "yearly_mean_demand.csv", fig_path= cfg.REPORT_DIR / "yearly_mean_demand.png",
    xlabel="Year",
    ylabel="Mean Demand (MW)",
    title="Yearly Mean Electricity Demand"
    ):
    # Generate resampled demand features
    Sample_freq_df= Sample_freq_df = generate_resampled_demand_features(
        df, resample_freq=resample_freq, aggregation_func=aggregation_func,
        rolling_window=rolling_window, start_date=start_date, end_date=end_date
    )

    # Plot and save 
    plot_and_save( 
        Sample_freq_df,
        resample_freq, 
        rolling_window, 
        csv_path, fig_path,
        xlabel,
        ylabel,
        title
        )
    return Sample_freq_df

######################################################################
# Plot monthly boxplot of electricity demand.
# df: DataFrame with columns 'Local time' and 'Demand'
#start_date : str, optional e.g. '2015-07-01'.
#end_date : str, optional e.g. '2026-05-31'.
#fig_path : str or Path, optional
#######################################################################    

def plot_monthly_demand_boxplot(
    df,
    start_date="2015-07-01",
    end_date="2026-05-31",
    fig_path= cfg.REPORT_DIR / "monthly_demand_boxplot.png"
):
  
    df = df.copy()

    # Convert datetime
    df['Local time'] = pd.to_datetime(df['Local time'])

    # Convert Demand to numeric
    df['Demand'] = pd.to_numeric(
        df['Demand'].astype(str).str.replace(',', '', regex=False)
    )

    # Filter date range
    if start_date:
        df = df[df['Local time'] >= pd.to_datetime(start_date)]

    if end_date:
        df = df[df['Local time'] <= pd.to_datetime(end_date)]

    # Month names
    df['month'] = df['Local time'].dt.month_name()

    # Month order
    month_order = [
        'January', 'February', 'March', 'April',
        'May', 'June', 'July', 'August',
        'September', 'October', 'November', 'December'
    ]

    df['month'] = pd.Categorical(
        df['month'],
        categories=month_order,
        ordered=True
    )

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))

    # Example 
    '''
    ax.plot(
        df['Local time'],
        df['Demand']
    )
    '''
    # Boxplot of demand by month
    df.boxplot(
        column='Demand',
        by='month',
        ax=ax
    )

    ax.set_title("Electricity Demand Distribution by Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Demand (MW)")
    ax.grid(True)

    # Remove automatic pandas title
    plt.suptitle("")
     
    plt.tight_layout()

    # Save figure
    if fig_path:
        plt.savefig(
            fig_path,
            format="svg",
            bbox_inches="tight"
        )

    plt.show()

######################################################################
# Plot monthly boxplot of electricity demand.
# df: DataFrame with columns 'Local time' and 'Demand'
#start_date : str, optional e.g. '2015-07-01'.
#end_date : str, optional e.g. '2026-05-31'.
#fig_path : str or Path, optional
#######################################################################    

def plot_daily_demand_boxplot(
    df,
    start_date="2015-07-01",
    end_date="2026-05-31",
    fig_path=   cfg.REPORT_DIR / "daily_demand_boxplot.png"
):

    df = df.copy()

    # Convert datetime
    df['Local time'] = pd.to_datetime(df['Local time'])

    # Convert Demand to numeric
    df['Demand'] = pd.to_numeric(
        df['Demand'].astype(str).str.replace(',', '', regex=False)
    )

    # Filter date range
    if start_date:
        df = df[df['Local time'] >= pd.to_datetime(start_date)]

    if end_date:
        df = df[df['Local time'] <= pd.to_datetime(end_date)]

    # Weekday names
    df['weekday'] = df['Local time'].dt.day_name()

    # Ensure correct weekday order
    weekday_order = [
        'Monday',
        'Tuesday',
        'Wednesday',
        'Thursday',
        'Friday',
        'Saturday',
        'Sunday'
    ]

    df['weekday'] = pd.Categorical(
        df['weekday'],
        categories=weekday_order,
        ordered=True
    )

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))

    df.boxplot(
        column='Demand',
        by='weekday',
        ax=ax
    )

    ax.set_title("Electricity Demand Distribution by day")
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Demand (MW)")
    ax.grid(True)

    # Remove automatic pandas title
    plt.suptitle("")

    plt.tight_layout()

    # Save figure
    if fig_path:
        plt.savefig(
            fig_path,
            format="svg",
            bbox_inches="tight"
        )

    plt.show()

######################################################################
# Plot daily boxplot of electricity demand.
# df: DataFrame with columns 'Local time' and 'Demand'
#start_date : str, optional e.g. '2015-07-01'.
#end_date : str, optional e.g. '2026-05-31'.
#fig_path : str or Path, optional
#######################################################################    

def plot_hourly_demand_boxplot(
    df,
    start_date="2015-07-01",
    end_date="2026-05-31",
    fig_path=cfg.REPORT_DIR / "hourly_demand_boxplot.png"
):

    df = df.copy()

    # Convert datetime
    df['Local time'] = pd.to_datetime(df['Local time'])

    # Convert Demand to numeric
    df['Demand'] = pd.to_numeric(
        df['Demand'].astype(str).str.replace(',', '', regex=False)
    )

    # Filter date range
    if start_date:
        df = df[df['Local time'] >= pd.to_datetime(start_date)]

    if end_date:
        df = df[df['Local time'] <= pd.to_datetime(end_date)]

    # Hour of day must be a column for pandas DataFrame.boxplot(group by).
    df['hour'] = df['Local time'].dt.hour

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))

    df.boxplot(
        column='Demand',
        by='hour',
        ax=ax
    )

    ax.set_title("Electricity Demand Distribution by Hour")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Demand (MW)")
    ax.grid(True)

    # Remove automatic pandas title
    plt.suptitle("")

    plt.tight_layout()

    # Save figure
    if fig_path:
        plt.savefig(
            fig_path,
            bbox_inches="tight"
        )

    plt.show()

def plot_mean_demand_by_hour_season(df, 
    fig_path=cfg.REPORT_DIR / "plot_mean_demand_by_hour_season.png"
                                    ):
    df = df.copy()

    # Convert datetime
    df['Local time'] = pd.to_datetime(df['Local time'])

    # Convert Demand to numeric
    df['Demand'] = pd.to_numeric(
        df['Demand'].astype(str).str.replace(',', '', regex=False)
    )

    # Extract hour and month
    df['hour'] = df['Local time'].dt.hour
    df['month'] = df['Local time'].dt.month

    df['season'] = df['month'].apply(get_season)

    # Mean demand by hour and season
    hourly_season_mean = (
        df.groupby(['hour', 'season'])['Demand']
          .mean()
          .reset_index()
    )

    # Pivot table: rows = hour, columns = season
    pivot_df = hourly_season_mean.pivot(
        index='hour',
        columns='season',
        values='Demand'
    )

    # Correct season order
    season_order = ['Spring', 'Summer', 'Fall', 'Winter']
    pivot_df = pivot_df[season_order]

    # Plot grouped bar chart
    ax = pivot_df.plot(
        kind='bar',
        figsize=(16, 6),
        width=0.85
    )

    plt.title('Mean Electricity Demand by Hour and Season')
    plt.xlabel('Hour of Day')
    plt.ylabel('Mean Demand (MW)')
    plt.xticks(rotation=0)
    plt.grid(axis='y', alpha=0.3)
    plt.legend(title='Season')
    plt.tight_layout()

    # Save figure
    if fig_path:
        plt.savefig(
            fig_path,
            bbox_inches="tight"
        )

    plt.show()

    return pivot_df 
#######################################################################    
#######################################################################    

def plot_mean_demand_by_weekday_season(df,
fig_path=cfg.REPORT_DIR / "plot_mean_demand_by_weekday_season.png"
                                    ):

    df = df.copy()

    # Convert datetime
    df['Local time'] = pd.to_datetime(df['Local time'])

    # Convert Demand to numeric
    df['Demand'] = pd.to_numeric(
        df['Demand'].astype(str).str.replace(',', '', regex=False)
    )

    # Weekday
    df['weekday'] = df['Local time'].dt.day_name()

    weekday_order = [
        'Monday',
        'Tuesday',
        'Wednesday',
        'Thursday',
        'Friday',
        'Saturday',
        'Sunday'
    ]

    # Month
    df['month'] = df['Local time'].dt.month

    df['season'] = df['month'].apply(get_season)

    # Mean demand
    weekday_season_mean = (
        df.groupby(['weekday', 'season'])['Demand']
          .mean()
          .reset_index()
    )

    # Pivot
    pivot_df = weekday_season_mean.pivot(
        index='weekday',
        columns='season',
        values='Demand'
    )

    # Order
    pivot_df = pivot_df.reindex(weekday_order)

    season_order = ['Spring', 'Summer', 'Fall', 'Winter']
    pivot_df = pivot_df[season_order]

    # Plot
    ax = pivot_df.plot(
        kind='bar',
        figsize=(12,6),
        width=0.8
    )

    plt.title(
        'Mean Electricity Demand by Weekday and Season'
    )
    plt.xlabel('Day of Week')
    plt.ylabel('Mean Demand (MW)')
    plt.xticks(rotation=0)
    plt.grid(axis='y', alpha=0.3)
    plt.legend(title='Season')
    plt.tight_layout()

    # Save figure
    if fig_path:
        plt.savefig(
            fig_path,
            bbox_inches="tight"
        )

    plt.show()

    return pivot_df

#######################################################################    
####################################################################### 

def plot_mean_demand_by_holiday(df, 
        fig_path=cfg.REPORT_DIR / "plot_mean_demand_by_holiday.png"):

    df = df.copy()

    # Convert datetime
    df['Local time'] = pd.to_datetime(df['Local time'])

    # Convert Demand to numeric
    df['Demand'] = pd.to_numeric(
        df['Demand'].astype(str).str.replace(',', '', regex=False)
    )

    # Extract month
    df['month'] = df['Local time'].dt.month
    df['month_name'] = df['Local time'].dt.month_name()

    # Identify holiday rows
    if df['holiday'].dtype == bool:
        holiday_mask = df['holiday']
        df['holiday_name'] = df['holiday'].map({True: 'Holiday'})
    else:
        holiday_text = df['holiday'].astype(str).str.strip()
        non_holiday_values = [
            '',
            '0',
            'false',
            'nan',
            'none',
            'no holiday',
            'not holiday'
        ]
        holiday_mask = (
            df['holiday'].notna()
            & ~holiday_text.str.lower().isin(non_holiday_values)
        )
        df['holiday_name'] = holiday_text

    holiday_df = df[holiday_mask].copy()
    non_holiday_df = df[~holiday_mask].copy()

    # Mean demand for each holiday in its corresponding month
    holiday_mean = (
        holiday_df.groupby(['holiday_name', 'month', 'month_name'])['Demand']
          .mean()
          .reset_index(name='Holiday Mean Demand')
    )

    # Mean demand of the same month, excluding holidays
    monthly_non_holiday_mean = (
        non_holiday_df.groupby('month')['Demand']
          .mean()
          .reset_index(name='Non-Holiday Monthly Mean Demand')
    )

    comparison_df = holiday_mean.merge(
        monthly_non_holiday_mean,
        on='month',
        how='left'
    )

    comparison_df['holiday_month'] = (
        comparison_df['holiday_name']
        + ' ('
        + comparison_df['month_name']
        + ')'
    )

    comparison_df = comparison_df.sort_values(
        ['month', 'holiday_name']
    )

    plot_df = comparison_df.set_index('holiday_month')[
        [
            'Holiday Mean Demand',
            'Non-Holiday Monthly Mean Demand'
        ]
    ]

    # Plot grouped bar chart
    ax = plot_df.plot(
        kind='bar',
        figsize=(14, 6),
        width=0.8
    )

    plt.title(
        'Mean Electricity Demand on Holidays vs Non-Holiday Monthly Mean'
    )
    plt.xlabel('Holiday')
    plt.ylabel('Mean Demand (MW)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.legend(title='Demand Type')
    plt.tight_layout()
    # Save figure
    if fig_path:
        plt.savefig(
            fig_path,
            bbox_inches="tight"
        )
    plt.show()

    return comparison_df

#######################################################################    
####################################################################### 

def plot_mean_demand_by_month(df, 
        fig_path=cfg.REPORT_DIR / "plot_mean_demand_by_month.png"):

    df = df.copy()

    # Convert datetime
    df['Local time'] = pd.to_datetime(df['Local time'])

    # Convert Demand to numeric
    df['Demand'] = pd.to_numeric(
        df['Demand'].astype(str).str.replace(',', '', regex=False)
    )

    # Extract month
    df['month'] = df['Local time'].dt.month

    # Mean demand by month
    monthly_mean = (
        df.groupby('month')['Demand']
          .mean()
          .reset_index()
    )

    # Month names
    month_names = [
        'Jan', 'Feb', 'Mar', 'Apr',
        'May', 'Jun', 'Jul', 'Aug',
        'Sep', 'Oct', 'Nov', 'Dec'
    ]

    monthly_mean['month_name'] = month_names

    # Plot
    plt.figure(figsize=(10, 5))

    plt.bar(
        monthly_mean['month_name'],
        monthly_mean['Demand']
    )

    plt.title('Mean Electricity Demand by Month')
    plt.xlabel('Month')
    plt.ylabel('Mean Demand (MW)')
    plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    # Save figure
    if fig_path:
        plt.savefig(
            fig_path,
            bbox_inches="tight"
        )
    plt.show()

    return monthly_mean

#######################################################################    
####################################################################### 
# Weather impact Analysis
#######################################################################    
####################################################################### 

def plot_scatter_temperature_vs_demand(
    df,
    start_date="2015-07-01",
    end_date="2026-05-31",
    gridsize=40,
    fig_path= cfg.REPORT_DIR / "temperature_vs_demand.png"
):
    df = df.copy()

    
    temperature_col = "temperature_2m"

    # Convert datetime
    df['Local time'] = pd.to_datetime(df['Local time'])

    # Convert Demand to numeric
    df['Demand'] = pd.to_numeric(
        df['Demand'].astype(str).str.replace(',', '', regex=False)
    )

    # Convert temperature to numeric
    df[temperature_col] = pd.to_numeric(
        df[temperature_col],
        errors='coerce'
    )

    # Remove missing values
    df = df.dropna(subset=[temperature_col, 'Demand'])

    # Filter date range
    if start_date:
        df = df[df['Local time'] >= pd.to_datetime(start_date)]

    if end_date:
        df = df[df['Local time'] <= pd.to_datetime(end_date)]

    # Create plot
    plt.figure(figsize=(8, 6))

    hb = plt.hexbin(
        df[temperature_col],
        df['Demand'],
        gridsize=gridsize,
        cmap='viridis'
    )

    plt.colorbar(hb, label='Number of Observations')

    plt.title('Temperature vs Electricity Demand Density')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Demand (MW)')
    plt.grid(True)

    if fig_path:
        plt.savefig(
            fig_path,
            dpi=300,
            bbox_inches='tight'
        )

    plt.show()


#######################################################################    
####################################################################### 

def plot_mean_demand_by_temperature_range(
    df,
    start_date="2015-07-01",
    end_date="2026-05-31",
    n_bins=20,
    fig_path= cfg.REPORT_DIR / "mean_demand_by_temperature_range.png"
):
    df = df.copy()

    # Convert datetime
    df['Local time'] = pd.to_datetime(df['Local time'])


    # Convert Demand to numeric
    df['Demand'] = pd.to_numeric(
        df['Demand'].astype(str).str.replace(',', '', regex=False)
    )

    # Convert temperature to numeric
    df['temperature_2m'] = pd.to_numeric(
        df['temperature_2m'],
        errors='coerce'
    )

    # Remove missing values
    df = df.dropna(subset=['temperature_2m', 'Demand'])

    # Filter date range
    if start_date:
        df = df[df['Local time'] >= pd.to_datetime(start_date)]

    if end_date:
        df = df[df['Local time'] <= pd.to_datetime(end_date)]

    # Create temperature bins
    df['temp_bin'] = pd.cut(
        df['temperature_2m'],
        bins=n_bins
    )

    # Mean demand for each temperature range
    temp_demand = (
        df.groupby(
            'temp_bin',
            observed=False
        )['Demand']
        .mean()
    )

    # Plot
    plt.figure(figsize=(12, 5))

    temp_demand.plot(
        kind='bar'
    )

    plt.xlabel('Temperature Range (°C)')
    plt.ylabel('Mean Demand (MW)')
    plt.title('Mean Electricity Demand by Temperature Range')
    plt.grid(True, axis='y')

    plt.tight_layout()

    if fig_path:
        plt.savefig(
            fig_path,
            dpi=300,
            bbox_inches='tight'
        )

    plt.show()


def plot_mean_demand_by_weather_range(
    df,
    weather_col,
    bin_col,
    xlabel,
    title,
    start_date="2015-07-01",
    end_date="2026-05-31",
    n_bins=20,
    fig_path= cfg.REPORT_DIR / f"mean_demand_by_weather_range.png"
):
    df = df.copy()

    # Convert datetime
    df['Local time'] = pd.to_datetime(df['Local time'])

    # Convert Demand to numeric
    df['Demand'] = pd.to_numeric(
        df['Demand'].astype(str).str.replace(',', '', regex=False)
    )

    # Convert weather feature to numeric
    df[weather_col] = pd.to_numeric(
        df[weather_col],
        errors='coerce'
    )

    # Remove missing values
    df = df.dropna(subset=[weather_col, 'Demand'])

    # Filter date range
    if start_date:
        df = df[df['Local time'] >= pd.to_datetime(start_date)]

    if end_date:
        df = df[df['Local time'] <= pd.to_datetime(end_date)]

    # Create weather feature bins
    df[bin_col] = pd.cut(
        df[weather_col],
        bins=n_bins
    )

    # Mean demand for each weather feature range
    weather_demand = (
        df.groupby(
            bin_col,
            observed=False
        )['Demand']
        .mean()
    )

    # Plot
    plt.figure(figsize=(12, 5))

    weather_demand.plot(
        kind='bar'
    )

    plt.xlabel(xlabel)
    plt.ylabel('Mean Demand (MW)')
    plt.title(title)
    plt.grid(True, axis='y')

    plt.tight_layout()

    if fig_path:
        plt.savefig(
            fig_path,
            dpi=300,
            bbox_inches='tight'
        )

    plt.show()

    return weather_demand


#######################################################################    
####################################################################### 

def plot_mean_demand_by_humidity_range(
    df,
    start_date="2015-07-01",
    end_date="2026-05-31",
    n_bins=20,
    fig_path=cfg.REPORT_DIR / "mean_demand_by_humidity_range.png"
):
    return plot_mean_demand_by_weather_range(
        df=df,
        weather_col='relative_humidity_2m',
        bin_col='humidity_bin',
        xlabel='Relative Humidity Range (%)',
        title='Mean Electricity Demand by Humidity Range',
        start_date=start_date,
        end_date=end_date,
        n_bins=n_bins,
        fig_path=fig_path
    )


#######################################################################    
####################################################################### 

def plot_mean_demand_by_apparent_temperature_range(
    df,
    start_date="2015-07-01",
    end_date="2026-05-31",
    n_bins=20,
    fig_path=cfg.REPORT_DIR / "mean_demand_by_apparent_temperature_range.png"
):
    return plot_mean_demand_by_weather_range(
        df=df,
        weather_col='apparent_temperature',
        bin_col='apparent_temp_bin',
        xlabel='Apparent Temperature Range (C)',
        title='Mean Electricity Demand by Apparent Temperature Range',
        start_date=start_date,
        end_date=end_date,
        n_bins=n_bins,
        fig_path=fig_path
    )


#######################################################################    
####################################################################### 

def plot_mean_demand_by_wind_speed_range(
    df,
    start_date="2015-07-01",
    end_date="2026-05-31",
    n_bins=20,
    fig_path=cfg.REPORT_DIR / "mean_demand_by_wind_speed_range.png"
):
    return plot_mean_demand_by_weather_range(
        df=df,
        weather_col='wind_speed_10m',
        bin_col='wind_speed_bin',
        xlabel='Wind Speed Range (km/h)',
        title='Mean Electricity Demand by Wind Speed Range',
        start_date=start_date,
        end_date=end_date,
        n_bins=n_bins,
        fig_path=fig_path
    )


#######################################################################    
####################################################################### 

def plot_mean_demand_by_snowfall_range(
    df,
    start_date="2015-07-01",
    end_date="2026-05-31",
    n_bins=20,
    fig_path=cfg.REPORT_DIR / "mean_demand_by_snowfall_range.png"
):
    return plot_mean_demand_by_weather_range(
        df=df,
        weather_col='snowfall',
        bin_col='snowfall_bin',
        xlabel='Snowfall Range (cm)',
        title='Mean Electricity Demand by Snowfall Range',
        start_date=start_date,
        end_date=end_date,
        n_bins=n_bins,
        fig_path=fig_path
    )


#######################################################################    
####################################################################### 

def plot_correlation_heatmap(
    df,
    columns=cfg.CORRELATION_HEATMAP_COLUMNS,
    fig_path=cfg.REPORT_DIR / "correlation_heatmap.png"
):
    df = df.copy()

    # Select columns
    if columns is None:
        corr_df = df.select_dtypes(include='number')
    else:
        corr_df = df[columns].copy()

    # Convert selected columns to numeric before calculating correlation.
    for col in corr_df.columns:
        corr_df[col] = pd.to_numeric(
            corr_df[col].astype(str).str.replace(',', '', regex=False),
            errors='coerce'
        )

    corr_df = corr_df.dropna(axis=1, how='all')

    # Correlation matrix
    corr_matrix = corr_df.corr()

    # Plot
    plt.figure(figsize=(10, 8))

    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5
    )

    plt.title("Correlation Heatmap")
    plt.tight_layout()

    if fig_path:
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")

    plt.show()



if __name__ == "__main__":

    # Load the electricity demand data
    elec_demand_df = dc.read_csv_file(cfg.REPORT_DIR / "all_features_dataset.csv")
    elec_demand_df["Local time"] = pd.to_datetime(elec_demand_df["Local time"])

    #plot_daily_mean_demand(df=elec_demand_df)
    #plot_monthly_mean_demand(df=elec_demand_df)
    #plot_yearly_mean_demand(df=elec_demand_df)
    #plot_monthly_demand_boxplot(elec_demand_df)
    #plot_mean_demand_by_month(elec_demand_df)
    #plot_daily_demand_boxplot(elec_demand_df)
    #plot_mean_demand_by_weekday_season(elec_demand_df)
    #plot_hourly_demand_boxplot(elec_demand_df)
    #plot_mean_demand_by_hour_season(elec_demand_df)
    #plot_mean_demand_by_holiday(elec_demand_df)
    #plot_mean_demand_by_holiday(elec_demand_df)
    ############
    ##########
    #plot_scatter_temperature_vs_demand(elec_demand_df)  # do not need
    #plot_mean_demand_by_temperature_range(elec_demand_df)
    #plot_mean_demand_by_humidity_range(elec_demand_df)
    #plot_mean_demand_by_wind_speed_range(elec_demand_df)
    #plot_mean_demand_by_snowfall_range(elec_demand_df)
    #plot_mean_demand_by_apparent_temperature_range(elec_demand_df)
    plot_correlation_heatmap(elec_demand_df) # not much useful information
