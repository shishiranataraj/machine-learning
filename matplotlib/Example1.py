import matplotlib.pyplot as plt

# To draw a line

x = [i for i in range(10)]
print(x)

y = [2*i for i in range(10)]
print(y)

plt.plot(x, y, color="red")
plt.title("Title")
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.show()