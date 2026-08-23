from dataclasses import dataclass
from itertools import combinations, product
from typing import Any, Dict, Iterable, List, Optional, Tuple, TypeVar
from abc import ABC, abstractmethod
from typing import TypeVar

__all__ = ["KCoverageResult", "KProjectionCoverage"]

Self = TypeVar("Self")

class Metric(ABC):
    def __init__(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def update(self: Self, *args, **kwargs) -> Self:
        """
        Add values to the metric
        """
        raise NotImplementedError

    @abstractmethod
    def compute(self, *args, **kwargs):
        """
        Compute the metric
        """
        raise NotImplementedError


class Projection:
    """
    A single projection
    """

    def __init__(self, n_values: List[int], names: Optional[List[str]] = None) -> None:
        """
        Holds a subspace with the cartesian product of some dimension and
        monitors coverage for this subspace.

        :param n_values: list with number of values for each dimension in this projection
        :param names: name of dimensions of this projection. just for debugging
        """
        self.n_options = n_values
        self.names = names or []

        # counts hits for all points in subspace
        self.counts: Dict[Tuple[int], int] = {}

        for point in product(*[range(i) for i in n_values]):
            self.counts[point] = 0

    @property
    def points(self) -> Iterable[Tuple[int]]:
        """
        :returns: iterator over all points in this subspace
        """
        return self.counts.keys()

    def is_covered(self, point: Tuple[int]) -> bool:
        """
        Checks of the particular point has been covered.
        We consider it covered if it has been covered at least once.
        """
        return self.counts[point] > 0

    @property
    def n_covered(self) -> int:
        """
        number of covered points in this subspace
        """
        return sum([int(count > 0) for count in self.counts.values()])

    @property
    def n_points(self) -> None:
        """
        number of points in this subspace
        """
        return len(self.counts)

    def reset(self) -> None:
        for key in self.counts:
            self.counts[key] = 0

    def cover(self, point: Tuple[int]) -> None:
        """
        mark point as covered
        """
        self.counts[point] += 1

    def __repr__(self) -> str:
        if self.names:
            return f'Projection({",".join(self.names)})'
        else:
            return "Projection()"

    @property
    def k(self) -> int:
        return len(self.n_options)


@dataclass
class KCoverageResult:
    """
    Result of k-projection coverage calculation.

    Args:
        coverage: The proportion of covered points over the total points (a float between 0 and 1).
        k: The number of dimensions used in the projection.
        covered: The number of points that are covered by the added scenarios.
        total: The total number of points in the k-dimensional space.
        scenes: The total number of scenarios that have been added.
    """

    coverage: float
    k: int
    covered: int
    total: int
    scenes: int


class KProjectionCoverage(Metric):
    """
    Preliminary implementation of Quantitative Projection Coverage from the paper
    [Quantitative Projection Coverage for Testing ML-enabled Autonomous Systems](https://arxiv.org/abs/1805.04333).

    **Note: This implementation is still missing some features, such as weighting.**

    Examples:
        How to use this metric:

            description = {
                "weather": ["good", "bad", "ugly"],
                "temperature": [1, 2, 3, 4],
                "humidity": [0.1, 0.2, 0.3, 0.4, 0.5],
            }

            cov = KProjectionCoverage(k=2, desc=description)

            cov.add_scenario({"weather": "bad", "temperature": 1, "humidity": 0.1})
            cov.add_scenario({"weather": "ugly", "temperature": 2, "humidity": 0.1})
            cov.add_scenario({"weather": "good", "temperature": 2, "humidity": 0.1})
            cov.add_scenario({"weather": "good", "temperature": 3, "humidity": 0.1})
            cov.add_scenario({"weather": "good", "temperature": 4, "humidity": 0.1})

            print(cov.compute())
    """

    def __init__(self, k: int, desc: Dict[str, List[Any]]) -> None:
        """
        Initializes the KProjectionCoverage metric.

        Args:
            k: The number of dimensions in the projection space (k <= total dimensions).
            desc: Domain description, a dictionary mapping each dimension to its list of possible values.
        """
        assert k <= len(desc)

        self.desc = desc
        self.k = k
        self.n_scenes = 0

        self.dims = list(desc.keys())
        self.n_dim_values = [len(desc[v]) for v in self.dims]

        # Maps (dimension, value) pairs to an integer index for efficiency.
        self.dim_value_to_index: Dict[Tuple[str, Any], int] = {}

        for d in self.dims:
            for n, o in enumerate(desc[d]):
                self.dim_value_to_index[(d, o)] = n

        # Projections identified by tuples of their corresponding dimensions.
        self.projections: Dict[Tuple[int], Projection] = {}

        # Create all k-dimensional projections
        for c in combinations(range(self.n_dims), r=self.k):
            self.projections[c] = Projection(
                n_values=[self.n_dim_values[j] for j in c],
                names=[self.dims[j] for j in c],
            )

    def reset(self) -> None:
        """
        Resets all projections, clearing the coverage data.
        """
        for p in self.projections.values():
            p.reset()

    @property
    def n_dims(self) -> int:
        """
        Returns the number of dimensions described by the domain.

        Returns:
            int: Number of dimensions.
        """
        return len(self.dims)

    def add_scenario(self, scenario: Dict[str, Any]) -> None:
        """
        Adds a new scenario to the coverage calculation.

        Args:
            scenario: A dictionary mapping each dimension to a value representing a scenario.
        """
        assert len(scenario) == self.n_dims

        # Convert the scenario description to its integer representation.
        scene = [self.dim_value_to_index[d, scenario[d]] for d in self.dims]

        # Update each k-dimensional projection with the new scenario.
        for c, projection in self.projections.items():
            point = tuple(scene[i] for i in c)
            projection.cover(point)

        self.n_scenes += 1

    def add_scenarios(self, scenarios: Iterable[Dict[str, Any]]) -> None:
        """
        Adds multiple scenarios to the coverage calculation.

        Args:
            scenarios: An iterable of dictionaries where each dictionary represents a scenario.
        """
        for scene in scenarios:
            self.add_scenario(scene)

    def update(self: Self, scenarios: Iterable[Dict[str, Any]]) -> Self:
        """
        Updates the coverage calculation with multiple scenarios.

        Args:
            scenarios: An iterable of dictionaries where each dictionary represents a scenario.

        Returns:
            Self: Returns the current instance to allow method chaining.
        """
        self.add_scenarios(scenarios=scenarios)
        return self

    def compute(self) -> KCoverageResult:
        """
        Computes the current coverage metrics based on the added scenarios.

        Returns:
            KCoverageResult: A result object containing coverage, total points, covered points, and the number of scenes.
        """
        covered = 0
        total = 0

        # Calculate total and covered points for each projection.
        for projection in self.projections.values():
            total += projection.n_points
            covered += projection.n_covered

        coverage = covered / total

        return KCoverageResult(
            coverage=coverage,
            k=self.k,
            covered=covered,
            total=total,
            scenes=self.n_scenes,
        )
