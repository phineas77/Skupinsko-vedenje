import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib

from fuzzy import triangular, linear, fprod, fsum, centroid, Rule, System, Value


class Sheep:
    def __init__(self, _id, x, y):
        self.id = _id
        self.x = x
        self.y = y
        self.theta = np.random.uniform(-np.pi, np.pi)
        self.speed = 1.
        self.turn = np.pi / 40

        heading = Value({'left': triangular(-np.pi, -np.pi / 2, 0),
                         'same': triangular(-np.pi / 2, 0, np.pi / 2),
                         'right': triangular(0, np.pi / 2, np.pi)})
        sig = Value({'low': linear(1, 0),
                     'high': linear(0, 1)})

        self.domain = np.linspace(-np.pi, np.pi)

        self.alignment = System([
            Rule(((sig.high, heading.left), min), (heading.left, fprod)),
            Rule(((sig.high, heading.right), min), (heading.right, fprod)),
            Rule(((sig.low, heading.same), max), (heading.same, fprod))
        ], fsum, self.domain)
        self.cohesion = System([
            Rule(((sig.high, heading.left), min), (heading.right, fprod)),
            Rule(((sig.high, heading.right), min), (heading.left, fprod)),
            Rule(((sig.low, heading.same), max), (heading.same, fprod))
        ], fsum, self.domain)
        self.separation = System([
            Rule(((sig.high, heading.left), min), (heading.left, fprod)),
            Rule(((sig.high, heading.right), min), (heading.right, fprod)),
            Rule(((sig.low, heading.same), max), (heading.same, fprod))
        ], fsum, self.domain)

    def distance(self, other: 'Sheep') -> float:
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def significance(self, other: 'Sheep', center: float = 25, slope: float = 5) -> float:
        return 1 - 1 / (1 + np.exp(-(self.distance(other) - center) / slope))

    def heading_angle(self, other: 'Sheep') -> float:
        return np.arctan2(other.y - self.y, other.x - self.x)

    def update(self):
        fuzzy_heading = []
        for agent in env.sheep:
            if agent.id == self.id:
                continue
            fuzzy_heading.append(self.alignment((agent.theta - self.theta,
                                                 self.significance(agent))))
            # fuzzy_heading.append(self.cohesion((self.theta - self.heading_angle(agent),
            #                                     self.significance(agent))))
            fuzzy_heading.append(self.separation((self.theta - self.heading_angle(agent),
                                                  self.significance(agent, 10, 1))))
        fuzzy_heading = np.array(fuzzy_heading).sum(axis=0)
        heading = centroid(fuzzy_heading, self.domain)
        return heading

    def step(self):
        theta = self.update()
        self.theta += np.clip(theta, -self.turn, self.turn)
        self.x = (self.x + self.speed * np.cos(self.theta)) % 200
        self.y = (self.y + self.speed * np.sin(self.theta)) % 200
        env.paths[self.id].append((self.x, self.y))


class Environment:
    def __init__(self):
        self.size = (200, 200)
        self.matrix = np.zeros(self.size)
        self.paths = {}

        def sample_points(size, num_points=20):
            sheep = []
            # np.random.seed(0)
            indices = np.random.choice(np.prod(size), num_points, replace=False)
            xs, ys = np.unravel_index(indices, size)
            for _id, (x, y) in enumerate(zip(xs, ys)):
                sheep.append(Sheep(_id, x, y))
                self.paths[_id] = []
            return sheep

        self.sheep: list[Sheep] = sample_points(self.size)

    def step(self):
        for sheep in self.sheep:
            sheep.step()

    def plot_env(self):
        for path in self.paths.values():
            x, y = zip(*path)
            plt.plot(x, y, alpha=0.5)
        plt.show()


if __name__ == '__main__':
    env = Environment()
    matplotlib.use('TkAgg')

    def update(_):
        env.step()
        updated_artists = []

        for sheep in env.sheep:
            path = env.paths[sheep.id]
            lines[sheep.id][0].set_data(*path[-1])
            tails[sheep.id][0].set_data([path[-1][0], path[-1][0] - 2 * np.cos(sheep.theta)],
                                        [path[-1][1], path[-1][1] - 2 * np.sin(sheep.theta)])
            updated_artists.extend(lines[sheep.id] + tails[sheep.id])

        return updated_artists

    steps = 1000
    fig, ax = plt.subplots()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 200)
    lines = [ax.plot([], [], '.') for _ in range(len(env.sheep))]
    tails = [ax.plot([], []) for _ in range(len(env.sheep))]
    ani = animation.FuncAnimation(fig, update, steps, interval=0, blit=True)
    # ani.save('sheep.gif', writer='ffmpeg', fps=30)
    plt.tight_layout()
    plt.show()

    # env.plot_env()
