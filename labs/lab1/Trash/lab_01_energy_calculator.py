# lab01_energy_calculator.ipynb (Python Equivalent)
# --------------------------------------------------
# CE49X: Introduction to Computational Thinking and Data Science
# Instructor: Dr. Eyuphan Koç | Fall 2025
# Author: Arben Üstün (2024706018)
# --------------------------------------------------
# This notebook calculates, classifies, and analyzes building energy consumption.
# All code cells executed and outputs shown.

# --------------------------------------------------
# EXERCISE 1: ENERGY CONSUMPTION BASICS
# --------------------------------------------------
print("\n=== Exercise 1: Energy Consumption Basics ===")

# --- Part A: Daily and Monthly Energy Calculation ---
lighting = 450  # kWh
hvac = 1200     # kWh
equipment = 350 # kWh
other = 180     # kWh

total_daily = lighting + hvac + equipment + other
monthly_consumption = total_daily * 30

print(f"Daily Energy Consumption:\nLighting: {lighting:.2f} kWh\nHVAC: {hvac:.2f} kWh\nEquipment: {equipment:.2f} kWh\nOther: {other:.2f} kWh\nTotal: {total_daily:.2f} kWh")
print(f"\nMonthly Consumption: {monthly_consumption:.2f} kWh")

# --- Part B: Unit Conversion ---
MJ = monthly_consumption * 3.6
GJ = MJ / 1000
print(f"\nMonthly Energy in MJ: {MJ:.2f} MJ")
print(f"Monthly Energy in GJ: {GJ:.2f} GJ")

# --- Part C: Cost Calculation ---
rate = 0.12
monthly_cost = monthly_consumption * rate
annual_cost = monthly_cost * 12
print(f"\nMonthly Cost: ${monthly_cost:,.2f}")
print(f"Annual Cost: ${annual_cost:,.2f}")

# --------------------------------------------------
# EXERCISE 2: BUILDING ENERGY ANALYSIS
# --------------------------------------------------
print("\n=== Exercise 2: Building Energy Analysis ===")

# Given Data
buildings = ['Office A', 'Retail B', 'School C', 'Hospital D', 'Apartment E']
monthly_consumption = [85000, 62000, 48000, 125000, 71000]
floor_area = [2500, 1800, 3200, 4000, 2800]

# --- Part A: Energy Intensity Calculation ---
energy_intensity = []
for i, building in enumerate(buildings):
    intensity = monthly_consumption[i] / floor_area[i]
    energy_intensity.append(intensity)
    print(f"{building}: {monthly_consumption[i]} kWh / {floor_area[i]} m^2 = {intensity:.2f} kWh/m^2/month")

# --- Part B: Statistical Analysis ---
total_consumption = sum(monthly_consumption)
average_consumption = total_consumption / len(monthly_consumption)
max_consumption = max(monthly_consumption)
min_consumption = min(monthly_consumption)

print(f"\nTotal Monthly Consumption: {total_consumption:,.2f} kWh")
print(f"Average Monthly Consumption: {average_consumption:,.2f} kWh")
print(f"Maximum Consumption: {max_consumption:,.2f} kWh")
print(f"Minimum Consumption: {min_consumption:,.2f} kWh")

# --- Part C: Find Buildings Above Average ---
above_avg_buildings = []
for i, building in enumerate(buildings):
    if monthly_consumption[i] > average_consumption:
        above_avg_buildings.append(building)

print(f"\nBuildings Above Average Consumption: {above_avg_buildings}")
print(f"Count: {len(above_avg_buildings)}")

# --------------------------------------------------
# EXERCISE 3: ENERGY EFFICIENCY CLASSIFIER
# --------------------------------------------------
print("\n=== Exercise 3: Energy Efficiency Classifier ===")

# --- Part A: Annual Energy Calculation ---
annual_intensity = [x * 12 for x in energy_intensity]
for i, building in enumerate(buildings):
    print(f"{building}: {annual_intensity[i]:.2f} kWh/m^2/year")

# --- Part B: Efficiency Classification ---
ratings = []
for intensity in annual_intensity:
    if intensity < 50:
        ratings.append('A')
    elif intensity < 100:
        ratings.append('B')
    elif intensity < 150:
        ratings.append('C')
    elif intensity < 200:
        ratings.append('D')
    else:
        ratings.append('F')

print("\n=== Energy Efficiency Report ===")
for i, building in enumerate(buildings):
    print(f"{building}: {annual_intensity[i]:.2f} kWh/m^2/year - Rating: {ratings[i]}")

# --- Part C: Rating Summary ---
print("\n=== Rating Summary ===")
for r in ['A', 'B', 'C', 'D', 'F']:
    print(f"Rating {r}: {ratings.count(r)} buildings")

most_common_rating = max(set(ratings), key=ratings.count)
print(f"Most Common Rating: {most_common_rating}")

# --------------------------------------------------
# EXERCISE 4: ENERGY COST CALCULATOR
# --------------------------------------------------
print("\n=== Exercise 4: Energy Cost Calculator ===")

# --- Part A: Simple Cost Function ---
def calculate_monthly_cost(consumption_kwh, rate_per_kwh):
    """Calculate monthly energy cost."""
    return consumption_kwh * rate_per_kwh

# Test function
cost_test = calculate_monthly_cost(50000, 0.12)
print(f"Monthly Cost for 50,000 kWh @ $0.12/kWh: ${cost_test:,.2f}")

# --- Part B: Peak/Off-Peak Cost Function ---
def calculate_tiered_cost(total_consumption, peak_percentage=0.6):
    peak_rate = 0.15
    off_peak_rate = 0.08
    peak_consumption = total_consumption * peak_percentage
    off_peak_consumption = total_consumption * (1 - peak_percentage)
    peak_cost = peak_consumption * peak_rate
    off_peak_cost = off_peak_consumption * off_peak_rate
    total_cost = peak_cost + off_peak_cost
    return peak_cost, off_peak_cost, total_cost

# Test function
peak_cost, off_peak_cost, total_cost = calculate_tiered_cost(85000, 0.6)
print(f"\nPeak Cost: ${peak_cost:,.2f}\nOff-Peak Cost: ${off_peak_cost:,.2f}\nTotal Cost: ${total_cost:,.2f}")

# --------------------------------------------------
# EXERCISE 5: ENERGY OPTIMIZATION
# --------------------------------------------------
print("\n=== Exercise 5: Energy Optimization ===")

# --- Part A: Most and Least Efficient Buildings ---
min_intensity = min(annual_intensity)
max_intensity = max(annual_intensity)
min_index = annual_intensity.index(min_intensity)
max_index = annual_intensity.index(max_intensity)

most_efficient = buildings[min_index]
least_efficient = buildings[max_index]
percentage_diff = ((max_intensity - min_intensity) / max_intensity) * 100

print(f"Most Efficient: {most_efficient} ({min_intensity:.2f} kWh/m^2/year, Rating {ratings[min_index]})")
print(f"Least Efficient: {least_efficient} ({max_intensity:.2f} kWh/m^2/year, Rating {ratings[max_index]})")
print(f"Percentage Difference: {percentage_diff:.2f}%")

# --- Part B: Energy Savings Potential ---
total_savings_kwh = 0
standard_rate = 0.12

print("\n=== Energy Savings Potential ===")
print("If all buildings achieved Rating B (100 kWh/m^2/year):")
for i, building in enumerate(buildings):
    if ratings[i] in ['C', 'D', 'F']:
        current_consumption = annual_intensity[i] * floor_area[i]
        target_consumption = 100 * floor_area[i]
        savings = current_consumption - target_consumption
        total_savings_kwh += savings
        print(f"{building}: Could save {savings:,.0f} kWh/year")

total_cost_savings = total_savings_kwh * standard_rate
print(f"\nTotal Potential Savings: {total_savings_kwh:,.0f} kWh/year")
print(f"Annual Cost Savings: ${total_cost_savings:,.2f}")

# --------------------------------------------------
# BONUS CHALLENGE: INTERACTIVE ENERGY CALCULATOR (+5 POINTS)
# --------------------------------------------------
print("\n=== Bonus Challenge: Interactive Energy Calculator ===")

while True:
    building_name = input("Enter building name (or 'quit' to exit): ")
    if building_name.lower() == 'quit':
        print("Exiting interactive calculator. Goodbye!")
        break
    try:
        monthly_kwh = float(input("Enter monthly consumption (kWh): "))
        floor_m2 = float(input("Enter floor area (m^2): "))
        energy_intensity = (monthly_kwh / floor_m2) * 12
        # Determine efficiency rating
        if energy_intensity < 50:
            rating = 'A'
        elif energy_intensity < 100:
            rating = 'B'
        elif energy_intensity < 150:
            rating = 'C'
        elif energy_intensity < 200:
            rating = 'D'
        else:
            rating = 'F'
        # Calculate monthly cost using tiered cost function
        _, _, total_cost = calculate_tiered_cost(monthly_kwh, 0.6)
        print(f"\n=== Analysis for {building_name} ===")
        print(f"Monthly Consumption: {monthly_kwh:.0f} kWh")
        print(f"Floor Area: {floor_m2:.0f} m²")
        print(f"Energy Intensity: {energy_intensity:.2f} kWh/m²/year")
        print(f"Efficiency Rating: {rating}")
        print(f"Monthly Cost (tiered): ${total_cost:,.2f}\n")
    except ValueError:
        print("Invalid input. Please enter numeric values for consumption and area.\n")