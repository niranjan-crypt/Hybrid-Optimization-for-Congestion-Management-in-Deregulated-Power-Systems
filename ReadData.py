import pandas as pd

def read_and_process_csv(file_path,file_path1):
    # Read the CSV file
    df = pd.read_csv(file_path)
    df1=pd.read_csv(file_path1)
    # Separate Load Bus Data and Shunt Capacitor Data
    load_bus_data = df
    generator_data = df1
    
    # Display Load Bus Data
    print("Load Bus Data:")
    print(load_bus_data)
    
    print("generator data:")
    print(generator_data)
    

    
    return load_bus_data

# Example usage
file_path = "D:\code\conf\load_bus_shunt_capacitor_data.csv"  # Replace with your file path
file_path1= "D:\code\conf\generator_data.csv"
load_bus= read_and_process_csv(file_path,file_path1)
