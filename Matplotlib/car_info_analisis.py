import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

car_data_dict = {
    'model': [
        'Nissan Skyline R34', 'Mazda RX-7', 'Toyota AE86', 'Nissan 240SX (S13)', 'Toyota Supra (MKIV)', 'Honda S2000', 'Mitsubishi Lancer Evolution (Evo VIII)', 'Subaru Impreza WRX STI (GC8)', 'Nissan Silvia (S15)', 'Mazda MX-5 Miata (NA)'
    ],
    'brand': [
        'Nissan', 'Mazda', 'Toyota', 'Nissan', 'Toyota', 'Honda', 'Mitsubishi', 'Subaru', 'Nissan', 'Mazda'
    ],
    'year': [
        1999, 2002, 1986, 1992, 1993, 1999, 2003, 1994, 1999, 1989
    ],
    'horsepower': [
        276, 276, 128, 140, 320, 240, 271, 276, 250, 116
    ],
    'top_speed': [
        249.448, 257.495, 201.168, 201.168, 284.852, 241.402, 249.448, 231.746, 225.308, 191.511
    ],
    'weight': [
        1560, 1250, 950, 1250, 1550, 1250, 1370, 1250, 1200, 980
    ],
    'width': [
        1780, 1755, 1625, 1700, 1810, 1755, 1770, 1690, 1695, 1675
    ],
    'heigth': [
        1360, 1230, 1335, 1285, 1265, 1270, 1455, 1415, 1285, 1235
    ],
    'length': [
        4500, 4280, 4280, 4440, 4520, 4115, 4350, 4355, 4280, 3975   
    ],
    'fuel_capacity': [
        70, 76, 50, 60, 75, 50, 55, 60, 55, 43
    ],
    'price': [
        50000, 35000, 25000, 15000, 60000, 25000, 25000, 20000, 18000, 12000
    ],
    'km_liter': [1.14, 1.06, 1.42, 1.08, 1.28, 1.4, 1.24, 1.16, 1.33, 1.44]
}

car_df = pd.DataFrame(car_data_dict)

print(car_df)


#MOST EXPENSIVE CAR
def get_expensive_car():
    plt.title('Car price (USD)')
    sns.barplot(x='model', y='price', data=car_df, hue='model');
    plt.show()



#MOST POWERFUL CAR
def get_powerful_car():
    plt.title('Car power (Horsepower)')
    sns.barplot(x='model', y='horsepower', data=car_df, hue='model');
    plt.show()



#TOP SPEEDY CAR
def get_top_speed():
    plt.title('Car top speed (Km/h)')
    sns.barplot(x='model', y='top_speed', data=car_df, hue='model')
    plt.show()



#GET WEIGHT-POWER RELATION
def get_weight_power_relation():
    car_df['weight_power'] = car_df.horsepower / car_df.weight
    plt.title('Car weight-power relation (hp/Kg)')
    sns.barplot(x='model', y='weight_power', data=car_df, hue='model')
    plt.show()


#CAR PERFORMANCE OVER THE YEARS
def get_car_performance_over_years():
    years_passed = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    degradation_rate = 5  # 5% degradation per year

    # Create an empty list to store DataFrames for each year
    performance_dfs = []

    # Loop through each year and calculate the updated horsepower
    for year in years_passed:
        # Create a copy of the original DataFrame to avoid modifying the original data
        car_df_copy = car_df.copy()
        
        # Calculate the updated horsepower for each car
        car_df_copy['horsepower'] = car_df_copy['horsepower'] * degradation_rate
        
        # Append the DataFrame for the current year to the list
        performance_dfs.append(car_df_copy)

    # Concatenate the DataFrames in the list to create one DataFrame
    performance_df = pd.concat(performance_dfs, keys=years_passed, names=['Year'])

    # Reset the index for the final DataFrame
    performance_df.reset_index(inplace=True)

    # Plot the performance over the years for each car
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='Year', y='horsepower', hue='model', data=performance_df)
    plt.title('Performance over the Years')
    plt.xlabel('Year')
    plt.ylabel('Horsepower')
    plt.legend(bbox_to_anchor=(0.80, 1), loc='upper left')
    plt.show()


def get_prices_distribution():
    plt.title('Prices distribution')
    plt.hist(car_df.price, color='purple');
    plt.show()


def get_weight_top_speed_coorelation():
    plt.title('Top speed - Weight coorelation')
    plt.xlabel('Top Speed (Km/h)')
    plt.ylabel('Weight (Kg)')
    sns.scatterplot(x='top_speed', y='weight', hue='model', data=car_df);
    plt.show()

def most_used_brand():
    brand_counts = car_df['brand'].value_counts()

    plt.figure(figsize=(10, 6))
    brand_counts.plot(kind='bar', color='skyblue')

    plt.title('Most used brand')
    plt.xlabel('Brand')
    plt.ylabel('Cuantity of cars')
    plt.xticks(rotation=45)

    plt.show()


def radar_graphic_width_heigth_length():

    cars = ['Nissan Skyline R34', 'Mazda RX-7', 'Toyota AE86', 'Nissan 240SX (S13)', 'Toyota Supra (MKIV)', 'Honda S2000', 'Mitsubishi Lancer Evolution (Evo VIII)', 'Subaru Impreza WRX STI (GC8)', 'Nissan Silvia (S15)', 'Mazda MX-5 Miata (NA)']
    width = [1780, 1755, 1625, 1700, 1810, 1755, 1770, 1690, 1695, 1675]
    heigth = [1360, 1230, 1335, 1285, 1265, 1270, 1455, 1415, 1285, 1235]
    length = [4500, 4280, 4280, 4440, 4520, 4115, 4350, 4355, 4280, 3975]

    # Create DataFrame with data.
    data = pd.DataFrame({
        'Car': cars,
        'Heigth': heigth,
        'Width': width,
        'Length': length
    })

    #Create radar chart
    categories = list(data.columns[1:]) #Variables to show in chart(extralling 'car')
    N = len(categories)

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close chart

    fig, ax = plt.subplots(subplot_kw={'polar': True}, figsize=(8, 8))

    for i, car in enumerate(cars):
        values = data.iloc[i, 1:].tolist()
        values += values[:1] #Close chart
        ax.fill(angles, values, alpha=0.90, label=car)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title('Cars width, heigth, length.')
    ax.legend(loc='lower right', bbox_to_anchor=(0.1, 0.1))

    plt.show()

def price_year_relation():
    plt.title('Price-Year relation')
    plt.xlabel('Price')
    plt.ylabel('Year')
    sns.scatterplot(x = 'price', y = 'year', hue='model', palette='magma', data=car_df) #Palettes: plasma, viridis, inferno, cividis, magma; coolwarm, RdBu_r, BrBG, PiYG; deep, pastel, colorblind, dark
    plt.show()

def km_per_liter():
    plt.title('Kilometers per liter relation')
    plt.xlabel('Km/L')
    plt.ylabel('Car')
    sns.barplot(x='km_liter', y='model', data=car_df)
    plt.show()

def fabrication_year_distribution():
    plt.title('Fabrication Year Distribution')
    plt.hist(car_df.year, color='purple')
    plt.show()






