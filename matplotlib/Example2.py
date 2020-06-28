from matplotlib import pyplot as plt

#To scatter points

x = [i for i in range(10)]
y = [2*i for i in range(10)]
plt.scatter(x, y, color="red")
plt.show()