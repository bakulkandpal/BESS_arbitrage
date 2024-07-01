This repository contains a simple Python code for a battery optimization model that strategically manages charging and discharging activities across various electricity markets with different time-granularity. The model aims to maximize profits while considering the impact of battery degradation over time. 

## **Dependencies**

The model relies on the following dependencies,

- **Pyomo**: An open-source package for formulating optimization problems.
- **BONMIN Solver**: An open-source solver capable of handling mixed integer nonlinear programming (MINLP) problems.

## **File Descriptions**

 - **model.py**: Main Python script containing the Pyomo model for BESS optimization.
 - **Results.xlsx**: Excel file that contains the results for each half-hour over three years with market revenue/costs and charging/discharging decisions.
