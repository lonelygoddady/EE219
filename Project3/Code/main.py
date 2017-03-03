import sys
import pro_1
import pro_3
import pro_4
import pro_5
import help_functions

def main(argv):
    table, movie_index = help_functions.fetch_valid_data()
    R = table.as_matrix()
    pro_1.pro_1(R)
    pro_3.pro_3()
    pro_4.pro_4(R,reg=False, reverse=False)
    pro_4.pro_4(R,reg=False, reverse=True)
    pro_4.pro_4(R,reg=True, reverse=False)
    pro_4.pro_4(R,reg=True, reverse=True)

    pro_5.pro_5(table)

if __name__ == "__main__":
    main(sys.argv)
