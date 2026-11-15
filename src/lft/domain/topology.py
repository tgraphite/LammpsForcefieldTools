"""Image-aware connectivity and bonded-term records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .structure import ImageShift


def _neg(shift: ImageShift) -> ImageShift:
    return tuple(-int(x) for x in shift)  # type: ignore[return-value]


def _sub(a: ImageShift, b: ImageShift) -> ImageShift:
    return tuple(int(x - y) for x, y in zip(a, b))  # type: ignore[return-value]


@dataclass(slots=True)
class Bond:
    i: int
    j: int
    order: float = 1.0
    image_shift: ImageShift = (0, 0, 0)
    source: str = "inferred"

    def __post_init__(self) -> None:
        self.image_shift = tuple(int(x) for x in self.image_shift)  # type: ignore[assignment]
        if self.i > self.j:
            self.i, self.j = self.j, self.i
            self.image_shift = _neg(self.image_shift)

    @property
    def key(self) -> tuple[int, int, ImageShift]:
        return (self.i, self.j, self.image_shift)

    def directed_shift(self, origin: int, target: int) -> ImageShift:
        if (origin, target) == (self.i, self.j):
            return self.image_shift
        if (origin, target) == (self.j, self.i):
            return _neg(self.image_shift)
        raise KeyError((origin, target))

    def to_dict(self) -> dict[str, Any]:
        return {
            "i": self.i,
            "j": self.j,
            "order": float(self.order),
            "image_shift": list(self.image_shift),
            "source": self.source,
        }


@dataclass(slots=True)
class LiftedTerm:
    atoms: tuple[int, ...]
    atom_images: tuple[ImageShift, ...]
    source: str = "enumerated"
    parameter_id: int | None = None

    kind = "term"

    def __post_init__(self) -> None:
        self.atoms = tuple(int(x) for x in self.atoms)
        self.atom_images = tuple(tuple(int(y) for y in image) for image in self.atom_images)  # type: ignore[assignment,misc]
        if len(self.atoms) != len(self.atom_images):
            raise ValueError("term atoms and atom_images have different lengths")

    @property
    def key(self) -> tuple[Any, ...]:
        return (self.atoms, self.atom_images)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "atoms": list(self.atoms),
            "atom_images": [list(image) for image in self.atom_images],
            "source": self.source,
            "parameter_id": self.parameter_id,
        }

    def reversed_about_second(self) -> "LiftedTerm":
        """Reverse a path and translate all images so the new second atom is image 0."""

        origin = self.atom_images[1]
        images = tuple(_sub(image, origin) for image in reversed(self.atom_images))
        return type(self)(tuple(reversed(self.atoms)), images, self.source, self.parameter_id)


@dataclass(slots=True)
class Angle(LiftedTerm):
    kind = "angle"


@dataclass(slots=True)
class Dihedral(LiftedTerm):
    kind = "dihedral"


@dataclass(slots=True)
class Improper(LiftedTerm):
    kind = "improper"


@dataclass(slots=True)
class Pair(LiftedTerm):
    kind = "pair"
