import help_functions
import numpy as np
import matplotlib.pyplot as plt

def pro_4(R, reg=False, reverse=False):
    W = np.sign(R)

    if reg is False:
        for k in [10, 50, 100]:
            print "now doing project 4 for k = {}".format(k)
            if reverse is False:
                U, V, total_LSE = help_functions.nmf(X=R, k=k, weight=W)  # switch R and W
            else:
                U, V, total_LSE = help_functions.nmf(X=W, k=k, weight=R)  # switch R and W
            
            prediction = np.dot(U, V) 
            print "************************************"
            print "total LSE for k = {0} is: {1}".format(k, total_LSE)
            print prediction
            print 'max is: ', np.max(prediction)
            
            if reverse is False:
                print R
                help_functions.ROC_curve_plotter(prediction, R, k, reverse=False)
            else:
                print W
                help_functions.ROC_curve_plotter(prediction, W, k, reverse=True)
            
        plt.show()

    else:
        for k in [10, 50, 100]:
            for i in [0.01, 0.1, 1]:
                print "now doing project 4 for k = " + str(k) + " and for lambda = " + str(i)
                if reverse is False:
                    U, V, total_LSE = help_functions.nmf(X=R, k=k, weight=W, reg_lambda=i)  # switch R and W
                else:
                    U, V, total_LSE = help_functions.nmf(X=W, k=k, weight=R, reg_lambda=i)  # switch R and W
                
                print "total LSE for k = {0} is: {1}".format(k, total_LSE)
                prediction = np.dot(U, V)
                print prediction
                
                if reverse is False:
                    print R
                    help_functions.ROC_curve_plotter(prediction, R, k, reg_lambda= i, reverse=False)
                else:
                    print W
                    help_functions.ROC_curve_plotter(prediction, W, k, reg_lambda= i, reverse=True)
            
        plt.show()

