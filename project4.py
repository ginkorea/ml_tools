import ann
import data
import numpy as np
import matplotlib.pyplot as plt

def project4(k, multiple=True):
    """function to perform the requirements for project 4"""
    databases = data.process(k)  # input data
    database = databases[0] # there is only one dataset
    network2 = ann.Network()  # create the 2 neuron neural network
    if multiple:  # used to determine best performance
        network3 = ann.Network(neurons_per_layer=3)
        network5 = ann.Network(neurons_per_layer=5)
        network10 = ann.Network(neurons_per_layer=10)
        networks = [network2, network3, network5, network10]
    else:  # use best performing network
        networks = [network2]
    errors = []
    weights = []
    for i in range(len(networks)):
        errors.append([])
    training_record_sets = []
    test_record_sets = []
    for i, training_set in enumerate(database.training_sets):  # prep the data
        training_records = ann.RecordSet(training_set, 0, 10)
        testing_records = ann.RecordSet(database.test_sets[i], 0, 10)
        training_record_sets.append(training_records)
        test_record_sets.append(testing_records)
    for i, training_record_set in enumerate(training_record_sets):  # train and test
        for j, network in enumerate(networks):
            error, trained = network.train_and_test_network(training_record_set, test_record_sets[i])
            errors[j].append(error)
            if not multiple:
                weight = network.return_input_layer_weights()
                weights.append(weight)
            network.reset_network()
    average_errors = []
    average_weights = []
    for row in errors:  # compute average errors
        average = np.mean(np.array(row))
        average_errors.append(average)
    print(average_errors)
    if not multiple:  # get weights
        for i in range(len(weights[0])):
            this_weight = []
            for row in weights:
                this_weight.append(row[i])
            this_weight = np.array([this_weight])
            this_weight = np.mean(this_weight)
            average_weights.append(this_weight)
        generate_weights_graph(average_weights)


def generate_weights_graph(average_weights):
    """function to generate the graph for the weights per input"""
    labels = ['BS', 'TK', 'SZ', 'SH', 'AD',
              'ES', 'BN', 'BC', 'NN', 'MS']

    plt.title("Figure 2: Breast Cancer Neural Network Input Weights")
    plt.xticks(range(len(labels)), labels)
    plt.bar(range(len(average_weights)), average_weights, width=.5, edgecolor="black", label=labels)
    plt.ylabel("Weight")
    plt.xlabel("Input Variables")
    plt.show()


def generate_results_graph():
    """function to generate the results from the multiple neural network width test"""
    d2 = 0.001500  # results from previous test
    d3 = 0.007214
    d5 = 0.008643
    d10 = 0.008643
    d = [d2, d3, d5, d10]
    labels = ['x 2', 'x 3', 'x 5', 'x 10']
    plt.title("Figure 1: Neural Network Accuracy by Width")
    plt.xticks(range(4), labels)
    plt.bar(range(len(d)), d, bottom=.04, width=.8, edgecolor="black", label=labels)
    plt.ylabel("Accuracy")
    plt.xlabel("Network Width")
    plt.show()


project4(5, multiple=False)  # runs the program; change to multiple=True for multiple ann
# generate_results_graph()