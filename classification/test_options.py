from options import BaseOptions, TrainOptions, TestOptions

opt = BaseOptions().parse()
print(opt)

opt_train = TrainOptions().parse()
print(opt_train)

opt_test = TestOptions().parse()
print(opt_test)
