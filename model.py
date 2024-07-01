import pandas as pd
from pyomo.environ import *
import matplotlib.pyplot as plt

parameters = pd.read_excel('Second Round Technical Question - Attachment 1.xlsx', sheet_name='Data', index_col=0)
half_hourly_data = pd.read_excel('Second Round Technical Question - Attachment 2.xlsx', sheet_name='Half-hourly data')
daily_data = pd.read_excel('Second Round Technical Question - Attachment 2.xlsx', sheet_name='Daily data')

max_charge_rate = parameters.loc['Max charging rate', 'Values']
max_discharge_rate = parameters.loc['Max discharging rate', 'Values']
max_storage_volume = parameters.loc['Max storage volume', 'Values']
eta_c = parameters.loc['Battery charging efficiency', 'Values']
eta_d = parameters.loc['Battery discharging efficiency', 'Values']
weight_deg = 1


model = ConcreteModel()
time_half = range(len(half_hourly_data))
days_total = range(len(daily_data))
model.charge1 = Var(time_half, domain=NonNegativeReals)
model.charge2 = Var(time_half, domain=NonNegativeReals)
model.charge3 = Var(time_half, domain=NonNegativeReals)
model.discharge1 = Var(time_half, domain=NonNegativeReals)
model.discharge2 = Var(time_half, domain=NonNegativeReals)
model.discharge3 = Var(time_half, domain=NonNegativeReals)
model.storage = Var(time_half, domain=NonNegativeReals, bounds=(0.2, max_storage_volume))  # The lowest SoC of BESS is limited to 20% to reduce degradation.
model.binary = Var(time_half, within=Binary)
model.half_hourly_rev = Var(time_half, domain=Reals)
model.delta = Var(time_half, within=NonNegativeReals)


model.profit = Objective(expr=sum(model.discharge1[t] * half_hourly_data.iloc[t]['Market 1 Price [£/MWh]'] - model.charge1[t]*half_hourly_data.iloc[t]['Market 1 Price [£/MWh]'] + model.discharge2[t]*half_hourly_data.iloc[t]['Market 2 Price [£/MWh]'] -
        model.charge2[t] * half_hourly_data.iloc[t]['Market 2 Price [£/MWh]'] + model.discharge3[t] * daily_data.iloc[t // 48]['Market 3 Price [£/MWh]'] - model.charge3[t]*daily_data.iloc[t // 48]['Market 3 Price [£/MWh]'] for t in time_half) - weight_deg*sum(model.delta[t] for t in time_half), sense=maximize)


model.constraints = ConstraintList()

for t in time_half:
    # 'Big-M' constraint with binary variables to limit charging/discharging at the same time across markets. 
    model.constraints.add(model.charge1[t] + model.charge2[t] + model.charge3[t] <= (1 - model.binary[t]) * 100)
    model.constraints.add(model.discharge1[t] + model.discharge2[t] + model.discharge3[t] <= model.binary[t] * 100)
    
    
    model.constraints.add(model.charge1[t] <= max_charge_rate)
    model.constraints.add(model.charge2[t] <= max_charge_rate)
    model.constraints.add(model.charge3[t] <= max_charge_rate)
    model.constraints.add(model.discharge1[t] <= max_discharge_rate)
    model.constraints.add(model.discharge2[t] <= max_discharge_rate)
    model.constraints.add(model.discharge3[t] <= max_discharge_rate)
    
    if t == 0:
        model.constraints.add(model.storage[t] == max_storage_volume/2)
    else:
        model.constraints.add(model.storage[t] == model.storage[t-1] + (model.charge1[t] + model.charge2[t] + model.charge3[t])*(1-eta_c) - (model.discharge1[t] + model.discharge2[t] + model.discharge3[t])/(1-eta_d))
        
        # Tracking the transition in charging to discharging or vice-versa. For degradation model.
        model.constraints.add(model.delta[t] >= model.binary[t] - model.binary[t-1])
        model.constraints.add(model.delta[t] >= -(model.binary[t] - model.binary[t-1]))
        
    # Enforcing the constraint that in Market 3 BESS has to charge/discharge for the entire day at same value.    
    if t//48 == (t-1)//48:
       model.constraints.add(model.charge3[t] == model.charge3[t-1])
       model.constraints.add(model.discharge3[t] == model.discharge3[t-1])
       
    # Saving each half-hour's cost/revenue from BESS scheduling as a separate variable model.half_hourly_rev   
    model.constraints.add(model.half_hourly_rev[t] == model.discharge1[t] * half_hourly_data.iloc[t]['Market 1 Price [£/MWh]'] - model.charge1[t]*half_hourly_data.iloc[t]['Market 1 Price [£/MWh]'] + model.discharge2[t]*half_hourly_data.iloc[t]['Market 2 Price [£/MWh]'] -
            model.charge2[t]*half_hourly_data.iloc[t]['Market 2 Price [£/MWh]'] + model.discharge3[t]*daily_data.iloc[t // 48]['Market 3 Price [£/MWh]'] - model.charge3[t]*daily_data.iloc[t // 48]['Market 3 Price [£/MWh]'])   
        
        
solver = SolverFactory('bonmin')
solver.options['TimeLimit'] = 340  
results = solver.solve(model, tee=True)

charge_data1, charge_data2, charge_data3 = [], [], []
discharge_data1, discharge_data2, discharge_data3 = [], [], []  
storage_data, half_hourly_econ = [], []  
binary_vals, cycle_binary = [], []


for t in time_half:
    charge_data1.append(model.charge1[t].value)
    charge_data2.append(model.charge2[t].value)
    charge_data3.append(model.charge3[t].value)
    discharge_data1.append(model.discharge1[t].value)
    discharge_data2.append(model.discharge2[t].value)
    discharge_data3.append(model.discharge3[t].value)
    storage_data.append(model.storage[t].value)
    half_hourly_econ.append(model.half_hourly_rev[t].value*0.5)  # Multiplied with 0.5 as the charging/discharging is for half an hour and price is in per MWh.
    binary_vals.append(model.binary[t].value)

    cycle_binary.append(model.delta[t].value)
    
data_export = {'Charging1': charge_data1, 'Charging2': charge_data2, 'Charging3': charge_data3,'Discharging1': discharge_data1, 'Discharging2': discharge_data2, 'Discharging3': discharge_data3, 'Market': half_hourly_econ}    
df = pd.DataFrame(data_export)
df.to_excel('exported_resuts.xlsx', index=False)

last_time_index=480
########## Figure for charging, discharging schedules of BESS. Total number of time slots to be shown in plot can be chosen as slicing indices below:
plt.figure(figsize=(14, 10))
plt.subplot(3, 1, 1) 
plt.plot(time_half[0:last_time_index], charge_data1[0:last_time_index], label='Market 1', color='blue')
plt.title('BESS Charging Schedule', fontsize=15)
plt.ylabel('Power [MW]', fontsize=12)
plt.legend()
plt.subplot(3, 1, 2)
plt.plot(time_half[0:last_time_index], charge_data2[0:last_time_index], label='Market 2', color='green')
plt.ylabel('Power [MW]', fontsize=12)
plt.legend()
plt.subplot(3, 1, 3)
plt.plot(time_half[0:last_time_index], charge_data3[0:last_time_index], label='Market 3', color='red')
plt.xlabel('Time')
plt.ylabel('Power [MW]', fontsize=12)
plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize=(14, 10))
plt.subplot(3, 1, 1)  
plt.plot(time_half[0:last_time_index], discharge_data1[0:last_time_index], label='Market 1', color='red')
plt.title('BESS Discharging Schedule', fontsize=15)
plt.ylabel('Power [MW]', fontsize=12)
plt.legend()
plt.subplot(3, 1, 2)
plt.plot(time_half[0:last_time_index], discharge_data2[0:last_time_index], label='Market 2', color='green')
plt.ylabel('Power [MW]', fontsize=12)
plt.legend()
plt.subplot(3, 1, 3)
plt.plot(time_half[0:last_time_index], discharge_data3[0:last_time_index], label='Market 3', color='purple')
plt.xlabel('Time')
plt.ylabel('Power [MW]', fontsize=12)
plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 5))
plt.plot(time_half[0:last_time_index], storage_data[0:last_time_index], color='purple')#, linestyle='--')
plt.xlabel('Time', fontsize=12)
plt.ylabel('Energy [MWh]', fontsize=12)
plt.title('SoC of BESS')
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(time_half[0:last_time_index],half_hourly_econ[0:last_time_index])
plt.xlabel('Time', fontsize=12)
plt.ylabel('Revenue/Costs [£]', fontsize=12)
plt.title('Half-hourly revenue/cost in all Markets (Revenue = +, Costs = -)')
plt.show()

