import help_functions
import numpy as np


def pro_1(R):
    for k in [10, 50, 100]:
        print "************************************"
        print "now doing proiect 1 for k = {}".format(k)
        U, V, total_LSE = help_functions.nmf(X=R, k=k)
        print "************************************"
        # print "matrix W is: "
        # print U
        # print "matrix H is: "
        # print V
        print "predict R is: "
        print np.dot(U, V)
        print "Original matrix is: "
        print R
        print "total LSE for k = {} is: {}".format(k, total_LSE)
        # print "\n\n"
        # print "************************************"
        # print "now doing proiect 1 for k = {} with buildin".format(k)
        # nmfmat = NMF(n_components=k, init='nndsvd', random_state=42)
        # U = nmfmat.fit_transform(R)
        # V = nmfmat.components_
        # print "************************************"
        # print "matrix W is: "
        # print U
        # print "matrix H is: "
        # print V
        # real_error = (R - np.dot(U,V))
        # print "real error sum is: "
        # print np.sum(real_error)
        # total_residual = np.sqrt(np.sum(real_error ** 2))
        # print "total LSE for k = {0} is: {1}".format(k, total_residual)
        print "\n\n\n\n\n"

R, movie_index = help_functions.fetch_valid_data()
pro_1(R)