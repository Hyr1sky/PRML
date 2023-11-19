import argparse

from p01b_logreg import main as p01b
from p01e_gda import main as p01e

parser = argparse.ArgumentParser()
parser.add_argument('p_num', nargs='?', type=int, default=0,
                    help='Problem number to run, 0 for all problems.')
args = parser.parse_args()

# Problem 1
if args.p_num == 0 or args.p_num == 1:
     
     """
     p01b(train_path='D:/VScode WorkStation/PRML/Task 2/data/ds1_train.csv',
          eval_path='D:/VScode WorkStation/PRML/Task 2/data/ds1_valid.csv',
          pred_path='D:/VScode WorkStation/PRML/Task 2/src/output/p01b_pred_1.txt')

     p01b(train_path='D:/VScode WorkStation/PRML/Task 2/data/ds2_train.csv',
          eval_path='D:/VScode WorkStation/PRML/Task 2/data/ds2_valid.csv',
          pred_path='D:/VScode WorkStation/PRML/Task 2/src/output/p01b_pred_2.txt')
     """
     p01b(train_path='./Task_2/data/ds1_train.csv',
          eval_path='./Task_2/data/ds1_valid.csv',
          pred_path='./Task_2/src/output/p01b_pred_1.txt')

     p01b(train_path='./Task_2/data/ds2_train.csv',
          eval_path='./Task_2/data/ds2_valid.csv',
          pred_path='./Task_2/src/output/p01b_pred_2.txt')
     
     p01e(train_path='./Task_2/data/ds1_train.csv',
          eval_path='./Task_2/data/ds1_valid.csv',
          pred_path='./Task_2/src/output/p01e_pred_1.txt')

     p01e(train_path='./Task_2/data/ds2_train.csv',
          eval_path='./Task_2/data/ds2_valid.csv',
          pred_path='./Task_2/src/output/p01e_pred_2.txt')