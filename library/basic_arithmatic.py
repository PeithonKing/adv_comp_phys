from copy import deepcopy as copy

def sum_of_AP(n, a, d):
    """Calculate the sum of the Arithmetic Progression upto N terms.

    Args:
        n (int): the number of terms in the AP
        a (float): the first term in the AP.
        d (float): the difference between the terms in the AP.

    Returns:
        float: Sum of the AP upto N terms.
    """
    return n * (2 * a + (n - 1) * d) / 2


def factorial(x):
    """Calculate the factorial of x.

    Args:
        x (int): the number whose factorial is to be calculated.

    Returns:
        int: Factorial of x.
    """
    if x == 0:
        return 1
    else:
        return x * factorial(x - 1)


def distance(a, b):
    """Calculate the distance between two points on N dimension list.

    Args:
        a (list): the first point.
        b (list): the second point. length of a and b must be same.

    Raises:
        ValueError: if the length of a and b are not same.

    Returns:
        float: the distance between the two points.
    """
    if len(a) != len(b):
        raise ValueError("a and b must be of same length")
    s = [(a[i]-b[i])**2 for i in range(len(a))]
    return sum(s)**0.5


def RMS(x):
    """Calculate the Root Mean Square of the list x.

    Args:
        x (list): the list whose RMS is to be calculated.

    Returns:
        float: the RMS of the list x.
    """
    return (sum([i**2 for i in x])/len(x))**0.5


class Complex:
    """A Class for Complex Numbers"""

    def __init__(
        self,
        real: float = 0,
        imag: float = 0,
        name: str = None,
        precision: int = 16
    ):
        """A complex number.

        Args:
            real (float): The real part of the complex number. Defaults to 0.
            imag (float): The imaginary part of the complex number. Defaults to 0.
            name (str, optional): The name of the Complex number; used while printing. Defaults to "complex".
            precision (int, optional): Number of digits to be printed after the decimal point. Defaults to 16.
        """
        self.real = real
        self.imag = imag
        self.name = name
        self.precision = precision

    def add(self, complex_2):
        """A function to return the sum of this complex and complex_2.

        Args:
            complex_2 (Complex): The complex number to be added to this complex number.

        Returns:
            Complex: The sum of this complex and complex_2. """
        return Complex(
            self.real + complex_2.real,
            self.imag + complex_2.imag,
            name=f"{self.name}+{complex_2.name}",
            precision=self.precision
        )

    def multiply(self, complex_2):
        """A function to return the product of this complex and complex_2

        Args:
            complex_2 (Complex): A complex number.

        Returns:
            Complex: A complex number representing the product of this complex and complex_2.
        """
        return Complex(
            self.real * complex_2.real - self.imag * complex_2.imag,
            self.real * complex_2.imag + self.imag * complex_2.real,
            name=f"{self.name}*{complex_2.name}",
            precision=self.precision
        )

    def mod(self):
        """mod value of this complex number.

        Returns:
            float: mod value of this complex number.
        """
        return (self.real ** 2 + self.imag ** 2)**0.5

    def __str__(self, name=True):
        """A function to return a string representation of this complex number."""
        a = ""
        if name and self.name:
            a += f"{self.name} = "
        if self.real:
            a += f"{round(self.real, self.precision)}"
            if self.imag:
                a += " + "
        if self.imag:
            a += f"{round(self.imag, self.precision)}i"
        if a == "":
            return "0 + 0i"
        if a == f"{self.name} = ":
            return f"{self.name} = 0 + 0i"
        return a
    
    def __add__(self, other):
        return self.add(other)
    
    def __mul__(self, other):
        return self.multiply(other)
    
    def __abs__(self):
        return self.mod()