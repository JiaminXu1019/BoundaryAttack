import os.path
import sys
import random
import numpy as np
from glob import glob
from natsort import natsorted
import argparse
from tensorflow.keras.models import load_model
from attacklib import *
from numpy.linalg import norm

def distance(v, u):
	return norm(v-u)

def parse_args():
	parser = argparse.ArgumentParser(description='Attack')
	parser.add_argument('--input_dir', '-input_dir', 
					  help='',
					  default='', type=str)
	parser.add_argument('--output_dir', '-output_dir', 
					  help='',
					  default='', type=str)
	parser.add_argument('--store_dir', '-store_dir', 
					  help='',
					  default='', type=str)
	parser.add_argument('--net', '-net', 
					  help='victim network path',
					  default='', type=str)
	parser.add_argument('--attack', '-attack', 
					  help='type of attack',
					  default='Boundary', type=str)
	parser.add_argument('--dataset', '-dataset', 
					  help='attack data',
					  default='ECG', type=str)
	parser.add_argument('--seed', dest='seed', help='a nonnegative integer for \
						reproducibility', default=0, type=int)
	parser.add_argument('--max_steps', dest='max_steps', help='Maximum number of iterations', 
						default=6700, type=int)
	parser.add_argument('--max_queries', dest='max_queries',
						help='Maximum number of queries', 
						default=150000, type=int)

	args = parser.parse_args()
	return args

args = parse_args()

labels = {
        0: "N",
        1: "S",
        2: "V",
        3: "F",
        4: "Q"
        }

def num_to_class(num):
        return(labels[num])

def main():
	# reproducibility 
	np.random.seed(args.seed)

    # load data in as original and target 
	if args.dataset == 'ECG':
			paths = glob(args.input_dir+"/*.npy")
			paths = natsorted(paths)
			o = np.load(paths[0])
			o = np.expand_dims(o, axis = 0)
			orig = np.int(o[0,-1,0])
			orig_ecg = o[:,:-1,:]
			t = np.load(paths[1])
			t = np.expand_dims(t, axis = 0)
			target = np.int(t[0,-1,0])
			target_ecg = t[:,:-1,:]

			# Each folder for attacking holds 2 files
			# One is the original ECG data
			# Other is taregt ECG data
			
			amodel = load_model(args.net)

	def predict_fn(ecg):
		return amodel.predict(ecg).argmax()

	if predict_fn(orig_ecg) != orig or predict_fn(target_ecg) != target:
		print('Image misclassified! No need to attack')
		sys.exit(0)
	else:
		print('Attack image of original class {}. Target class: {}'.format(num_to_class(orig), num_to_class(target))) 
			
	if args.attack == 'Boundary':
		attack = BoundaryAttack(amodel, shape=orig_ecg.shape, args=args)
		adversarial, queries = attack.attack(target_ecg, orig_ecg, orig, target, max_iterations=args.max_steps, max_queries=args.max_queries)
		print("Finished.")
		
	if adversarial is not None:
		np.save(args.store_dir, adversarial)
		print('Successfully found adversarial ECG with distance {:.5e} and {} queries'.format(distance(orig_ecg,adversarial), queries))	

if __name__ == '__main__':
	main()
