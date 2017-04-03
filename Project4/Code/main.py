import sys
import help_functions
import part_a
import part_b
import part_c
import part_d
import part_e
import part_f

def main(argv):
    data, X_tfidf, _, _ = help_functions.data_load()
    print("running part B")
    part_b.part_b(data, X_tfidf)
    print("running part C")
    part_c.part_c(data, X_tfidf)
    print("running part D")
    part_d.part_d(data, X_tfidf)
    print("running part E")
    part_e.part_e()
    print("running part F")
    part_f.part_f()

if __name__ == "__main__":
    main(sys.argv)
