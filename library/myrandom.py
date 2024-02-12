from tqdm import tqdm
import matplotlib.pyplot as plt
try:
    from library.basic_arithmatic import distance
except:
    from basic_arithmatic import distance

class Random:
    def __init__(self, seed: float = 0.1, range: list = [0,1]):
        """Random number generator.

        Args:
            seed (float, optional): Initial Seed. Defaults to 0.1.
        """
        self.seed = seed
        self.scale = lambda x: range[0] + x*(range[1]-range[0])

    def rand(self, c = 3.5):
        """A function to generate a random number by using the formula:
            X[n+1] = c * X[n] * (1- X[n]).
            
        Args:
            c (float, optional): Defaults to 3.5.
            
        Returns:
            float: a random number between 0 and 1.
        """
        self.seed = c*self.seed*(1-self.seed)
        return self.seed

    def LCG(self, a = 1103515245, c = 12345, m = 32768):  # LCG
        """
            A function to generate a random number using the Linear Congruential Generator algorithm.
        Formula Used: X[n+1] = (a*X[n] + c) mod m.
        Value is divided by m before returning for scaling between 0 and 1.

        Args:
            seed (float, optional): A value between 0 and 1. Seed is used if provided, else uses the last random value generated as seed.
            a (int, optional): Defaults to 1103515245.
            c (int, optional): Defaults to 12345.
            m (int, optional): Defaults to 32768.

        Returns:
            float: a random number between 0 and 1.
        """
        self.seed = (a*self.seed + c) % m
        return self.scale(self.seed / m)

    def walk(
            self,
            steps: int = 300,
            dimension: int = 2,
            plot: bool = True,
            colour: str = "#1f77b4"
        ):
        """A function to generate a random walk.

        Args:
            steps (int, optional): Number of steps. Defaults to 300.
            # seed (float, optional): Seed of the random walk. Defaults to 0.45.
            dimension (int, optional): Random walk in how many dimentional space. We can only plot if the walk is on 2D. Defaults to 2.
            plot (bool, optional): Whether to plot the walk or not. Defaults to True if dimensions is 2 else False.
            colour (str, optional): Colour of the path plotted. Defaults to "#1f77b4".

        Returns:
            points (list): A list of points in the walk.
            dist (list): A list of consecutive distances between the points.
        """
        if dimension != 2:
            plot = False

        points = [[0]*dimension]

        d = range(steps)
        if plot: d = tqdm(d)

        for i in d:
            point = [points[-1][i] + 2*self.LCG() - 1 for i in range(dimension)]
            # point = points[-1][:]  # a copy of the last point
            # for i in range(dimension):
            #     point[i] = point[i] + 2*self.LCG() - 1
            points.append(point)

        if plot:
            pointsx, pointsy = zip(*points)
            # Axes
            plt.plot([max(pointsx), min(pointsx)], [0, 0], '#000000')
            plt.plot([0, 0], [max(pointsy), min(pointsy)], '#000000')

            plt.plot(pointsx, pointsy, colour)  # The path of the random walk
            plt.plot(pointsx[0], pointsy[0], "go")  # starting point
            plt.plot(pointsx[-1], pointsy[-1], "ro")  # ending point
            plt.show()
        
        dist = [distance(points[i], points[i+1])**2 for i in range(len(points)-1)]
        
        return points, dist

# if __name__ == "__main__":
#     from tqdm import tqdm
#     from basic_arithmatic import distance
