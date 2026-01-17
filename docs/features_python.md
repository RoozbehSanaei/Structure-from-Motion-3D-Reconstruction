# Modern Python language features used in `templering_klt_ba_loop_slam_v3.py`

## Postponed evaluation of type annotations (`from __future__ import annotations`)
```python
from __future__ import annotations
```
This makes type annotations “lazy” (stored as strings until needed), which avoids forward-declaration problems and reduces import-order headaches. It is similar in spirit to being able to reference a type before it is fully defined, without needing a prior declaration. In this file, it helps because classes reference each other in annotations (e.g., `PoseCW.inv() -> "PoseWC"` and `@classmethod ... -> Self`) without forcing you to reorder code.

## Type aliases + NumPy array typing (`TypeAlias`, `NDArray`)
```python
from typing import TypeAlias, Self
from numpy.typing import NDArray

F64: TypeAlias = np.float64
Mat33: TypeAlias = NDArray[F64]
Vec3: TypeAlias = NDArray[F64]
Pts2: TypeAlias = NDArray[F32]  # (N,2)
```
This is a “typedef” layer for readability and correctness. Instead of seeing `NDArray[np.float64]` everywhere, the code uses domain names (`Mat33`, `Vec3`) that convey intent (3×3 rotation, 3D vector, 2D points). It is analogous to `using Mat33 = ...;` in C++ to make APIs self-documenting and to help tooling (type checkers, IDEs) catch mismatches.

## Dataclasses as “modern structs” (`@dataclass`) with `slots` and `frozen`
```python
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class PoseCW:
    R: Mat33
    t: Vec3
```
`@dataclass` auto-generates boilerplate you’d normally write in C++ for simple POD-like types (constructor, repr, comparisons where applicable). `slots=True` prevents dynamic attribute creation (like restricting the object layout), improving memory use and attribute access speed—closer to fixed-layout structs. `frozen=True` makes instances immutable after construction (conceptually similar to making members `const`), which is valuable for pose objects so they don’t get accidentally mutated across optimization steps.

## Named constructors and static factory methods (`@staticmethod`, `@classmethod`, `Self`)
```python
@dataclass(frozen=True, slots=True)
class PoseCW:
    @staticmethod
    def I() -> Self:
        return PoseCW(np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64))

    @staticmethod
    def from_rvec_t(rvec: Vec3, t: Vec3) -> Self:
        return PoseCW(rot_exp(rvec), np.asarray(t, dtype=np.float64).reshape(3))
```
This mirrors common C++ patterns like `static PoseCW Identity();` or `static PoseCW FromRodrigues(...)`. `Self` communicates “this method returns an instance of the current class,” which is especially useful if you subclass later. It avoids sprinkling the class name throughout return annotations and makes refactors safer.

## String-valued enums (`StrEnum`) for clean CLI / config
```python
from enum import StrEnum

class TranslationMode(StrEnum):
    FULL = "full"
    DIR  = "dir"
    ROT  = "rot"
```
This is similar to `enum class`, but with string values that integrate directly with CLI parsing and logging. Instead of manually mapping between strings and enum values, the enum itself is the string. It reduces “stringly-typed” code while keeping user-facing options readable (`--translation-mode full|dir|rot`).

## Pattern matching (`match/case`) as a structured `switch`
```python
match mode:
    case TranslationMode.ROT:
        return rR
    case TranslationMode.DIR:
        rt = (unit(t_pred) - unit(e.t_ji)) * e.w_trans
        return np.hstack([rR, rt])
    case TranslationMode.FULL:
        rt = (t_pred - e.t_ji) * e.w_trans
        return np.hstack([rR, rt])
```
This is Python’s modern equivalent of a `switch` with clearer semantics and fewer “fall-through” hazards. Here it cleanly selects different residual models (rotation-only vs translation-direction vs full translation), and the structure makes it hard to accidentally handle one mode incorrectly.

## Modern union types (`|`) instead of verbose `Optional[...]`
```python
self._prev_gray: NDArray[U8] | None = None

def pid_for(self, tid: int) -> int | None:
    return self.track_to_pid.get(tid)
```
This is syntactic modernization: `T | None` reads similarly to “nullable T” and avoids older `Optional[T]`. For a C++ reader, it is conceptually close to “this may be absent” (think `std::optional<T>`), even though Python does not enforce it at runtime without extra tooling.

## `pathlib.Path` everywhere (filesystem paths as objects)
```python
from pathlib import Path

ap.add_argument("--out", type=Path, default=Path("outputs_klt_ba_loop_v3"))

args.out.mkdir(parents=True, exist_ok=True)
write_ply_xyz(args.out / "templeRing_sparse_points.ply", xyz)
```
This is very similar to `std::filesystem::path`: paths are first-class objects that support safe joining (`/` operator), creation, traversal, etc. It avoids brittle string concatenation and makes IO code more portable and less error-prone.

## Safer container defaults (`field(default_factory=...)`)
```python
from dataclasses import field

@dataclass(slots=True)
class MapPoint:
    pid: int
    track_id: int
    Xw: Vec3
    obs: list[tuple[int, NDArray[F32]]] = field(default_factory=list)
```
This prevents the classic “shared default list” bug. In C++ terms, it is the difference between each instance having its own `std::vector` vs accidentally using a single static vector across all instances. `default_factory=list` guarantees a fresh list per `MapPoint`.

## Efficient grouping (`defaultdict`) for track histories
```python
from collections import defaultdict

self.track_hist: dict[int, list[tuple[int, NDArray[F32]]]] = defaultdict(list)
...
self.track_hist[tid].append((kf_id, uv))
```
`defaultdict(list)` auto-initializes missing keys with empty lists. That removes repeated “if key not in map: create container” logic. It is similar to using `unordered_map<int, vector<...>>` with `operator[]` in C++, where accessing a missing key constructs a default value.

## High-level CLI flags with `BooleanOptionalAction`
```python
ap.add_argument(
    "--visuals",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Write montage/matches/trajectory/cloud images."
)
```
This gives you both `--visuals` and `--no-visuals` automatically, which is a modern ergonomic improvement over older “store_true only” patterns. It makes the CLI explicit and symmetric without extra code.

## Comprehensions for declarative transforms (compact, readable data shaping)
```python
obs = {int(tid): uv.copy() for tid, uv in zip(self.tracker.ids.tolist(), self.tracker.pts)}

centers = np.vstack([kf.pose_cw.t for kf in sys.keyframes]).astype(np.float64, copy=False)
```
These are concise “map” operations that resemble common C++ ranges pipelines in intent: build a new container by transforming elements from an existing one. Used carefully, they reduce boilerplate loops while keeping the dataflow obvious (e.g., observations built from tracked IDs + points; trajectory built from keyframe camera centers).

## Properties as computed accessors (`@property`)
```python
@dataclass(frozen=True, slots=True)
class MiddleburyRecord:
    img: str
    K: Mat33
    R: Mat33
    t: Vec3

    @property
    def pose_wc(self) -> PoseWC:
        return PoseWC(self.R, self.t)
```
This provides a method-like computation exposed as a field-like attribute, similar to a C++ getter (`PoseWC pose_wc() const;`) but accessed as `rec.pose_wc`. It keeps call sites clean while preserving encapsulation and allows the implementation to change without altering the public API.
