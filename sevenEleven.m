%%setup network
network = SupervisedHebbianLayer(30, "hardlims");

%input patterns
p0 = [-1 1 1 1 -1 1 -1 -1 -1 1 1 -1 -1 -1 1 1 -1 -1 -1 1 1 -1 -1 -1 1 -1 1 1 1 -1];
p1 = [-1 1 1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 -1];
p2 = [1 1 1 -1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 -1 1 1 -1 -1 -1 1 -1 -1 -1 -1 1 1 1 1];
p3 = [-1 1 1 1 -1 1 -1 -1 -1 1 -1 -1 1 1 1 -1 -1 -1 -1 1 1 -1 -1 -1 1 -1 1 1 1 -1];
p4 = [1 -1 -1 -1 1 1 -1 -1 -1 1 1 1 1 1 1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 -1 -1 -1 1];
p5 = [1 1 1 1 1 1 -1 -1 -1 -1 1 1 1 1 -1 -1 -1 -1 -1 1 1 -1 -1 -1 1 -1 1 1 1 -1];

%%train the network
network = network.train(p0', p0');
network = network.train(p1', p1');
network = network.train(p2', p2');
network = network.train(p3', p3');
network = network.train(p4', p4');
network = network.train(p5', p5');

%test the network
output = network.forward(p2);


%print for validation
disp("weights: ");
disp(network.weights);
SupervisedHebbianLayer.printOut(network.weights);
