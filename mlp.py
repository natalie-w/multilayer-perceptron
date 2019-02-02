
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt



class mlp:
	def __init__(self, inputs, targets, nhidden):
		self.beta = 1
		self.eta = 0.01
		self.momentum = 0.0
		self.nhidden = nhidden

	def earlystopping(self, inputs, targets, valid, validtargets):
		##stop when validation error starts increasing

		old_val_error1 = 100002
		old_val_error2 = 100001
		new_val_error = 100000

		count = 0

		## initialize all weights to small random values
		self.w1 = [[2/np.sqrt(self.nhidden) * np.random.random() - 1/np.sqrt(self.nhidden) for j in range(self.nhidden)] for i in range(40)]
		self.b1 = [2/np.sqrt(self.nhidden) * np.random.random() - 1/np.sqrt(self.nhidden) for i in range(self.nhidden)]
		self.w2 = [[2/np.sqrt(self.nhidden) * np.random.random() - 1/np.sqrt(self.nhidden) for j in range(8)] for i in range(self.nhidden)]
		self.b2 = [2/np.sqrt(self.nhidden) * np.random.random() - 1/np.sqrt(self.nhidden) for i in range(8)]

		while (((old_val_error1 - new_val_error) > 0.001) or ((old_val_error2 - old_val_error1) > 0.001)):
			self.train(inputs, targets, 10)
			old_val_error2 = old_val_error1
			old_val_error1 = new_val_error
			validout = self.forward(valid)
			new_val_error = 0.5 * np.sum((validtargets - validout)**2)
			##print('error', new_val_error)


	def train(self, inputs, targets, iterations=100):

		def dot_prod_plus_bias(input_vector, weight_matrix, bias_matrix):
			dot_prod = []
			weights = list(zip(*weight_matrix))
			for col in range(len(weight_matrix[0])):
				v_weights = weights[col]
				row = 0
				for e in range(len(input_vector)):
					row += input_vector[e] * v_weights[e]
				dot_prod.append(row - bias_matrix[col])
			return dot_prod

		def sigmoid(x):
			return [sigmoid_single(i) for i in x]

		def sigmoid_single(x):
			return 1/(1+np.exp(-self.beta * x))

		def derivative_sigmoid(x):
			return [i * (1 - i) for i in x]

		## training
		for it in range(iterations):

			##shuffle data
			inputs, targets = shuffle(inputs, targets)

			for v in range(len(inputs)):
				vector = inputs[v]
				target = targets[v]

				## forwardfeed
				
				# calculate hidden input with w1, b1    	
				hidden = dot_prod_plus_bias(vector, self.w1, self.b1)
				h_activation = sigmoid(hidden)
				# calculate output with w2, b2
				out = dot_prod_plus_bias(h_activation, self.w2, self.b2)

				## error
				output_error = [y - t for y, t in zip(out, target)]
				h_error = []

				for h in range(len(h_activation)):
					multiplier = h_activation[h] * (1 - h_activation[h])
					w_sum = 0
					for e in range(len(output_error)):
						o_error = output_error[e]
						weight = self.w2[h][e]
						w_sum += o_error * weight
					h_error.append(multiplier * w_sum)


				##update output layer weights
				for i in range(len(self.w2)):
					for j in range(len(self.w2[0])):
						self.w2[i][j] -= self.eta * output_error[j] * h_activation[i]

				##update hidden layer weights
				for k in range(len(self.w1)):
					for l in range(len(self.w1[0])):
						self.w1 -= self.eta * h_error[l] * vector[k]


				##update bias
				for i in range(len(self.b2)):
					self.b2[i] += self.eta * output_error[i]

				for j in range(len(self.b1)):
					self.b1[j] += self.eta * h_error[j]


	def forward(self, inputs):
		## returns result of putting inputs into current neural net
		
		def dot_prod_plus_bias(input_vector, weight_matrix, bias_matrix):
			dot_prod = []
			weights = list(zip(*weight_matrix))
			for col in range(len(weight_matrix[0])):
				v_weights = weights[col]
				row = 0
				for e in range(len(input_vector)):
					row += input_vector[e] * v_weights[e]
				dot_prod.append(row - bias_matrix[col])
			return dot_prod

		def sigmoid(x):
			return [sigmoid_single(i) for i in x]

		def sigmoid_single(x):
			return 1/(1+np.exp(-self.beta * x))

		def max_only(lst):
			curr = 0
			curr_max = lst[curr]
			final = [1]
			for i in range(1, len(lst)):
				if lst[i] > curr_max:
					curr_max = lst[i]
					final[curr] = 0
					final.append(1)
					curr = i
				else:
					final.append(0)
			return final


		output = []
		for i in range(len(inputs)):
			hidden = dot_prod_plus_bias(inputs[i], self.w1, self.b1)
			h_activation = sigmoid(hidden)
			out = dot_prod_plus_bias(h_activation, self.w2, self.b2)
			output.append(out)
		return output

	def confusion(self, inputs, targets):
		def to_num(lst):
			for i in range(len(lst)):
				if lst[i] == 1:
					return i

		def max_only(lst):
			curr = 0
			curr_max = lst[curr]
			final = [1]
			for i in range(1, len(lst)):
				if lst[i] > curr_max:
					curr_max = lst[i]
					final[curr] = 0
					final.append(1)
					curr = i
				else:
					final.append(0)
			return final


		s = self.forward(inputs)
		preds = [max_only(o) for o in s]
		y_pred = [to_num(el) for el in preds]
		y_test = [to_num(t) for t in targets]

		cnf_matrix = confusion_matrix(y_test, y_pred)
		
		print('hidden nodes: ', self.nhidden, ' accuracy_score:', accuracy_score(y_test, y_pred))
		print(cnf_matrix)
		plt.imshow(cnf_matrix)
		title = 'Confusion Matrix for ' + str(self.nhidden) + ' nodes'
		plt.title(title)
		tick_marks = np.arange(8)
		classes = np.arange(8)
		plt.xticks(tick_marks, classes)
		plt.yticks(tick_marks, classes)
		plt.ylabel('True label')
		plt.xlabel('Predicted label')
		plt.savefig('../' + title + '.png')






