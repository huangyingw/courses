from matplotlib import pyplot as plt
from sklearn.metrics import log_loss

# [Log Loss](http://wiki.fast.ai/index.php/Log_Loss) doesn't support probability values of 0 or 1--they are undefined (and we have many). Fortunately, Kaggle helps us by offsetting our 0s and 1s by a very small value. So if we upload our submission now we will have lots of .99999999 and .000000001 values. This seems good, right?
#
# Not so. There is an additional twist due to how log loss is
# calculated--log loss rewards predictions that are confident and correct
# (p=.9999,label=1), but it punishes predictions that are confident and
# wrong far more (p=.0001,label=1). See visualization below.

# Visualize Log Loss when True value = 1
# y-axis is log loss, x-axis is probabilty that label = 1
# As you can see Log Loss increases rapidly as we approach 0
# But increases slowly as our predicted probability gets closer to 1

x = [i * .0001 for i in range(1, 10000)]
y = [log_loss([1, 0], [i * .0001, 1 - (i * .0001)], eps=1e-15)
     for i in range(1, 10000, 1)]

plt.plot(x, y)
plt.axis([-.05, 1.1, -.8, 10])
plt.title("Log Loss when true label = 1")
plt.xlabel("predicted probability")
plt.ylabel("log loss")

plt.show()
