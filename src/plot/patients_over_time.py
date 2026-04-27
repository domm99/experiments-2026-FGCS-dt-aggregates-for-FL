import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_patient_growth(df_list):
    full_df = pd.concat([df[['Patient_ID', 'Measurement_date']] for df in df_list])

    full_df['Measurement_date'] = pd.to_datetime(full_df['Measurement_date'])
    patient_ranges = full_df.groupby('Patient_ID')['Measurement_date'].agg(['min', 'max']).reset_index()

    all_dates = pd.date_range(start=patient_ranges['min'].min(),
                              end=patient_ranges['max'].max(),
                              freq='D')

    sns.set_style('whitegrid')

    active_counts = []
    for date in all_dates:
        count = ((patient_ranges['min'] <= date) & (patient_ranges['max'] >= date)).sum()
        active_counts.append(count)

    plot_data = pd.DataFrame({'Date': all_dates, 'Active_Patients': active_counts})

    # 6. Plotting
    #sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))

    ax = sns.lineplot(data=plot_data, x='Date', y='Active_Patients',
                      drawstyle='steps-post', color='#2e7d32', linewidth=4)


    plt.fill_between(plot_data['Date'], plot_data['Active_Patients'], alpha=0.3, color='#2e7d32')

    for spine in plt.gca().spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.2)

    ax.tick_params(axis='both', which='major', labelsize=25)
    plt.title('', fontsize=15, pad=20)
    plt.xlabel('Date', fontsize=28)
    plt.ylabel('Active patients', fontsize=28)
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig('charts/patients-over-time.png')

if __name__ == '__main__':
    files = glob.glob('T1DiabetesGranada/split-labeled/*.csv')
    dfs = [pd.read_csv(file) for file in files]
    plot_patient_growth(dfs)