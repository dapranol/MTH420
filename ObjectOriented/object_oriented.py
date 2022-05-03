# object_oriented.py
"""Python Essentials: Object Oriented Programming.
<Lucia>
<MTH420>
<4/24/22>
"""


class Backpack:
    """A Backpack object class. Has a name and a list of contents.

    Attributes:
        name (str): the name of the backpack's owner.
        contents (list): the contents of the backpack.
    """

    # Problem 1: Modify __init__() and put(), and write dump().
    def __init__(self, name, color, max_size = 5):
        """Set the name and initialize an empty list of contents.

        Parameters:
            name (str): the name of the backpack's owner.
        """

        self.name = name
        self.contents = []
        self.color = []
        self.max_size = 5

    def put(self, item):
        """Add an item to the backpack's list of contents with the parameter that the max size of the backpack is 5 items"""
        self.contents.append(item)
        if max_size > 5:
            print("No Room!") 
        else:
            print()

    def dump(self):
        """Remove all the items in the backpack"""
        return self.contents.remove()

    def take(self, item):
        """Remove an item from the backpack's list of contents."""
        self.contents.remove(item)

    # Magic Methods -----------------------------------------------------------

    # Problem 3: Write __eq__() and __str__().
    def __add__(self, other):
        """Add the number of contents of each Backpack."""
        return len(self.contents) + len(other.contents)

    def __lt__(self, other):
        """Compare two backpacks. If 'self' has fewer contents
        than 'other', return True. Otherwise, return False.
        """
        return len(self.contents) < len(other.contents)


# An example of inheritance. You are not required to modify this class.
class Knapsack(Backpack):
    """A Knapsack object class. Inherits from the Backpack class.
    A knapsack is smaller than a backpack and can be tied closed.

    Attributes:
        name (str): the name of the knapsack's owner.
        color (str): the color of the knapsack.
        max_size (int): the maximum number of items that can fit inside.
        contents (list): the contents of the backpack.
        closed (bool): whether or not the knapsack is tied shut.
    """
    def __init__(self, name, color):
        """Use the Backpack constructor to initialize the name, color,
        and max_size attributes. A knapsack only holds 3 item by default.

        Parameters:
            name (str): the name of the knapsack's owner.
            color (str): the color of the knapsack.
            max_size (int): the maximum number of items that can fit inside.
        """
        Backpack.__init__(self, name, color, max_size=3)
        self.closed = True

    def put(self, item):
        """If the knapsack is untied, use the Backpack.put() method."""
        if self.closed:
            print("I'm closed!")
        else:
            Backpack.put(self, item)

    def take(self, item):
        """If the knapsack is untied, use the Backpack.take() method."""
        if self.closed:
            print("I'm closed!")
        else:
            Backpack.take(self, item)

    def weight(self):
        """Calculate the weight of the knapsack by counting the length of the
        string representations of each item in the contents list.
        """
        return sum(len(str(item)) for item in self.contents)


# Problem 2: Write a 'Jetpack' class that inherits from the 'Backpack' class.
class Jetpack(Backpack):
    def __init__(self, name, color, max_size = 2, fuel = 10):
        self.fuel = 10
        """change superclass to have additional arguments"""

    def fly(fuel_burned):
        fuel_burned = input("amount of fuel to be burned")
        if fuel_burned < fuel:
            print ("Not enough fuel!")
        else:
            print (fuel - fuel_burned)
        """ function that excepts an amount of fuel to be burned and decrements the fuel attribute by that amount"""
    def dump(self):
        return self.contents.remove()
        return self.fuel.remove()
        """empty the contents and the fuel tank"""

# Problem 4: Write a 'ComplexNumber' class.

if __name__ == "__main__":

#problem 1

    print __init__(self, name)
    print put(self, item)
    print def dump(self)
    print take(self, item)
