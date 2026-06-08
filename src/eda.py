import matplotlib.pyplot as plt
import pandas as pd
import data_collection as dc
import config as cfg


###################################
# Plot electricity demand over time
# df: DataFrame with columns 'Local time' and 'Demand'
# resample_freq:  'D', 'W', 'ME' and 'YE'
# aggregation_func: 'sum', 'mean', 'median', 'min' and 'max', 'std'
# start_date: 'YYYY-MM-DD'
# end_date:  'YYYY-MM-DD'
###################################
def plot_electricity_demand_over_time(
    df, resample_freq='D', aggregation_func='sum', rolling_window=30,
    start_date=None,
    end_date=None,
    csv_path=None, fig_path=None
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

    # Plotting
    plt.figure(figsize=(13,5))
    plt.plot(Sample_freq_df['Local time'], Sample_freq_df['Demand'])
    plt.plot(Sample_freq_df['Local time'], Sample_freq_df[f'Rolling_{rolling_window}d'], label=f'Rolling Mean ({rolling_window} periods)', color='orange')

    plt.title("Electricity Demand")
    plt.xlabel(f"Time ({resample_freq})")
    plt.ylabel(f"Demand ({aggregation_func})")
    plt.grid(True)

    # Save to CSV and png
    if csv_path:
        Sample_freq_df.to_csv(csv_path, index=False)
    else:
        Sample_freq_df.to_csv(cfg.REPORT_DIR / "electricity_demand.csv", index=False)

    if fig_path:
        plt.savefig(fig_path)
    else:
        plt.savefig(cfg.REPORT_DIR / "electricity_demand.svg", format='svg')

    plt.show()

    return Sample_freq_df

###################################
# Plot monthly boxplot of electricity demand.
# df: DataFrame with columns 'Local time' and 'Demand'
#start_date : str, optional e.g. '2015-07-01'.
#end_date : str, optional e.g. '2026-05-31'.
#fig_path : str or Path, optional
###################################
def plot_monthly_demand_boxplot(
    df,
    start_date=None,
    end_date=None,
    fig_path=None
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

    # Ensure correct month order
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

###################################
# Plot Weekday boxplot of electricity demand.
# df: DataFrame with columns 'Local time' and 'Demand'
#start_date : str, optional e.g. '2015-07-01'.
#end_date : str, optional e.g. '2026-05-31'.
#fig_path : str or Path, optional
###################################
def plot_weekday_demand_boxplot(
    df,
    start_date=None,
    end_date=None,
    fig_path=None
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

    ax.set_title("Electricity Demand Distribution by Weekday")
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

if __name__ == "__main__":

    # Load the electricity demand data
    elec_demand_df = dc.read_csv_file(cfg.REPORT_DIR / "all_features_dataset.csv", cols=["Local time", "Demand"])
    elec_demand_df["Local time"] = pd.to_datetime(elec_demand_df["Local time"])
    '''
    #  daily total electricity demand
    plot_electricity_demand_over_time(
        elec_demand_df,
        start_date="2016-01-01",
        end_date="2026-05-01", aggregation_func="sum", resample_freq='D', rolling_window=30,
        csv_path=cfg.REPORT_DIR / "daily_total_electricity_demand.csv",
        fig_path=cfg.REPORT_DIR / "daily_total_electricity_demand.png"
    )
    # The 10 max daily electricity demand, 

    '''
    '''
    # Mean daily electricity demand
    plot_electricity_demand_over_time(
        elec_demand_df,
        start_date="2016-01-01",
        end_date="2026-05-01", aggregation_func="mean", resample_freq='D', rolling_window=30,
        csv_path=cfg.REPORT_DIR / "daily_mean_electricity_demand.csv",
        fig_path=cfg.REPORT_DIR / "daily_mean_electricity_demand.png"
    ) 
    '''
    '''
    # Mean monthly electricity demand - seasonality
    plot_electricity_demand_over_time(
        elec_demand_df,
        start_date="2015-07-01",
        end_date="2026-05-01", aggregation_func="mean", resample_freq='M', rolling_window=10,
        csv_path=cfg.REPORT_DIR / "monthly_mean_electricity_demand.csv",
        fig_path=cfg.REPORT_DIR / "monthly_mean_electricity_demand.png"
    ) 
    '''
    # Plot monthly boxplot of electricity demand.

    plot_monthly_demand_boxplot(
        elec_demand_df, 
        start_date="2015-07-01", 
        end_date="2026-05-01", 
        fig_path=cfg.REPORT_DIR / "monthly_demand_boxplot.png"
        )
    # Plot weekday boxplot of electricity demand.

    plot_weekday_demand_boxplot(
        elec_demand_df, 
        start_date="2015-07-01", 
        end_date="2026-05-01", 
        fig_path=cfg.REPORT_DIR / "weekday_demand_boxplot.png"
        )

