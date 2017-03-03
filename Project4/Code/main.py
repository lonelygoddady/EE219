import sys
import help_functions
# import part_a
import part_b
# import part_c
# import part_d

def main(argv):
    data, X_tfidf, X_counts, count_vect = help_functions.data_load()
    part_b.part_b(data, X_tfidf)
    # part_c.part_c(data, X_tfidf)
    # part_d.part_d(data, X_tfidf)
if __name__ == "__main__":
    main(sys.argv)
