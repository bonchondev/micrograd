from typing import Any
import numpy as np
from sklearn.datasets import make_moons
from numpy._typing import NDArray
from sqlalchemy.orm import Session
from db import Base, SmallY, BigX, engine
import random


def init_data():
    np.random.seed(1337)
    random.seed(1337)

    moons = make_moons(n_samples=100, noise=0.1)

    X: NDArray[Any] | Any = moons[0]
    y: NDArray[np.intp] | Any = moons[1] 

    y = y*2 - 1 # make y be -1 or 1

    Base.metadata.create_all(engine)

    with Session(engine) as session:
        for point in X:
            x_point = BigX(first_value=point[0], second_value=point[1])
            session.add(x_point)
        for item in y:
            y_point = SmallY(value=item)
            session.add(y_point)
        session.commit()

    with open('big_x.txt', 'w') as f:
        for line in X:
            f.writelines([str(l) for l in line])
    with open('small_y.txt', 'w') as f:
        f.writelines(str(y))

init_data()
