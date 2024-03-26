from options import TrainOptions

opt = TrainOptions().parse()

from networks import define_network, define_criterion, define_optimizer, init_network

network = define_network(opt)
print(network)
print(network.state_dict().keys())

print(network.state_dict()["feature_extractor.0.weight"])
print(network.state_dict()["feature_extractor.0.bias"])
print(network.state_dict()["classifier.0.weight"])
print(network.state_dict()["classifier.0.bias"])

init_network(network)

print(network.state_dict()["feature_extractor.0.weight"])
print(network.state_dict()["feature_extractor.0.bias"])
print(network.state_dict()["classifier.0.weight"])
print(network.state_dict()["classifier.0.bias"])

criterion = define_criterion(opt)
print(criterion)

optimizer = define_optimizer(network, opt)
print(optimizer)