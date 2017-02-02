import helper
import sys
import numpy as np
import fileconverter

def main(argv):
    network_data = fileconverter.open_file('network_backup_dataset.csv')
    housing_data = fileconverter.open_file('housing_data.csv')

    network_dataT = np.transpose(network_data)
    housing_dataT = np.transpose(housing_data)
    
    target_network = network_dataT[5]
    
    target_housing = housing_dataT[len(housing_dataT)-1]
    features_housing = np.transpose(housing_dataT[0:len(housing_dataT)-1])

    features_network = np.zeros((len(network_data), 6))
    for x in range(len(network_data)):
        features_network[x][0] = network_data[x][0]
        features_network[x][1] = network_data[x][1]
        features_network[x][2] = network_data[x][2]
        features_network[x][3] = network_data[x][3]
        features_network[x][4] = network_data[x][4]
        features_network[x][5] = network_data[x][6]
    # print(len(target_network))
    helper.plot_generation(network_data)           
    helper.problem2a(features_network,target_network)
    helper.random_forest(features_network, target_network)                 # part 2b
    helper.neuralNetworkRegression(features_network, target_network)       # part 2c
    helper.problem3(features_network,target_network)
    helper.problem4()                                         # part 4
    helper.boston_housing_pr5(features_housing, target_housing)
    pass

if __name__ == "__main__":
    main(sys.argv)

 
    

