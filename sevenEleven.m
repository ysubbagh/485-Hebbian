%% setup network
network = SupervisedHebbianLayer(30, "hardlims");
pseudoNet = SupervisedHebbianLayer(30, "hardlims");

%input patterns
p0 = [-1 1 1 1 -1 1 -1 -1 -1 1 1 -1 -1 -1 1 1 -1 -1 -1 1 1 -1 -1 -1 1 -1 1 1 1 -1];
p1 = [-1 1 1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 -1];
p2 = [1 1 1 -1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 -1 1 1 -1 -1 -1 1 -1 -1 -1 -1 1 1 1 1];
p3 = [-1 1 1 1 -1 1 -1 -1 -1 1 -1 -1 1 1 1 -1 -1 -1 -1 1 1 -1 -1 -1 1 -1 1 1 1 -1];
p4 = [1 -1 -1 -1 1 1 -1 -1 -1 1 1 1 1 1 1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 -1 -1 -1 1];
p5 = [1 1 1 1 1 1 -1 -1 -1 -1 1 1 1 1 -1 -1 -1 -1 -1 1 1 -1 -1 -1 1 -1 1 1 1 -1];
P = [p0' p1' p2' p3' p4' p5'];

%% train the networks
network = network.train(p0', p0');
network = network.train(p1', p1');
network = network.train(p2', p2');
network = network.train(p3', p3');
network = network.train(p4', p4');
network = network.train(p5', p5');

%pseudoinverse train the function
pseudoNet = pseudoNet.pseudoInverseRule(P, P);

%% print for validation
SupervisedHebbianLayer.printNumbs(p0, 1, "0");
SupervisedHebbianLayer.printNumbs(p1, 2, "1");
SupervisedHebbianLayer.printNumbs(p2, 3, "2");
SupervisedHebbianLayer.printNumbs(p3, 4, "3");
SupervisedHebbianLayer.printNumbs(p4, 5, "4");
SupervisedHebbianLayer.printNumbs(p5, 6, "5");
SupervisedHebbianLayer.printWeights(network.weights, 7, "traditional training");
SupervisedHebbianLayer.printWeights(pseudoNet.weights, 8, "pseudoinverse training");


