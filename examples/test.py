from main import ChordProcessor, ApproximityType


def f(x: int | float) -> int | float:
    return x**3+3*x**2-1


e = 10 ** -3

processor = ChordProcessor(f, epsilon=e, interval_find_step=0.1, aprxm=ApproximityType.MIXED)

processor.find_solutions()
processor.print_last_equations()