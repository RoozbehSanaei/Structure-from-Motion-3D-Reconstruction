## Named choices with strings

### `StrEnum` for readable configuration values

`StrEnum` is a way to define a small set of named options where each option is a real text value. This keeps configuration readable, because the program can compare against clear names instead of remembering special strings scattered across the code. In `templering_sfm.py`, `TranslationMode` and `ExportGeometry` are defined as `StrEnum` types. That lets the program accept user-facing values like `"full"` or `"pointcloud"` while still working with a controlled list of allowed choices. Generic use is to centralize “allowed words” in one place. In this code, those enum values are used to drive branches for motion constraints and export behavior.

```python
from enum import StrEnum

class ExportGeometry(StrEnum):
    NONE = "none"
    POINTCLOUD = "pointcloud"
    BOTH = "both"

mode = ExportGeometry.POINTCLOUD
if mode == ExportGeometry.NONE:
    pass
```

---

## Object attributes that compute values

### `@property` for “looks like a field, runs like a function”

A `property` lets you expose a method as if it were a simple attribute. You access it without parentheses, which keeps calling code clean, while still allowing the value to be computed on demand. In `templering_sfm.py`, `MiddleburyRecord` defines `pose_wc` and `pose_cw` as properties. That means other parts of the program can write `rec.pose_cw` and receive a computed pose object, without needing to remember the conversion math every time. Generic use is to hide repeated calculations behind a friendly attribute name. Here, it packages “build the pose from stored R and t” into a single, readable access.

```python
class MiddleburyRecord:
    def __init__(self, R, t):
        self.R = R
        self.t = t

    @property
    def pose_wc(self):
        return (self.R, self.t)  # simplified: returns pose parts
```

---

## Methods that are tied to the class, not the instance

### `@staticmethod` for helpers that live inside a class

A `staticmethod` is a function stored inside a class for organization, but it does not receive the usual `self` object. This is useful when the logic belongs “near” the class conceptually, yet it does not need access to a particular instance. In `templering_sfm.py`, `TempleRing._read_par` and `TempleRing._read_ang` are static methods because they are parsing helpers: they take a path, read a file, and return structured data. Generic use is to keep related helpers grouped with the type they support. In this code, it prevents those file-reading utilities from floating around as unrelated global functions.

```python
class TempleRing:
    @staticmethod
    def _read_ang(path):
        # simplified: read and return a dictionary
        return {}
```

### `@classmethod` for alternate constructors that return `cls(...)`

A `classmethod` receives the class itself (commonly named `cls`). This is useful for “factory” functions that create an instance in a particular way. In `templering_sfm.py`, `TempleRing.from_zip` and `TempleRing.from_dir` are class methods: they prepare data, then return a fully built `TempleRing` object. Generic use is to offer multiple clean entry points for building the same type, while keeping the core initialization consistent. Here, it allows loading from either an extracted folder or a zip archive, without duplicating the object creation logic across the codebase.

```python
class TempleRing:
    def __init__(self, root, records, angles):
        self.root = root
        self.records = records
        self.angles = angles

    @classmethod
    def from_dir(cls, root):
        records = []   # simplified
        angles = {}
        return cls(root, records, angles)
```

---

## Numeric operations that read like math

### Matrix multiplication with the `@` operator

The `@` operator performs matrix multiplication. It is designed for math-heavy code so the intent stays obvious: `A @ B` reads like “multiply these two matrices.” In `templering_sfm.py`, it appears in camera projection and pose conversions, such as `K @ ...` when building a camera projection matrix and `R.T @ t` when moving between coordinate frames. Generic use is to keep linear-algebra expressions short and close to textbook form. In this code, it helps keep the camera model readable, which matters because those expressions are used repeatedly in tracking, triangulation, and optimization.

```python
import numpy as np

K = np.eye(3)
R = np.eye(3)
t = np.zeros((3, 1))

P = K @ np.hstack([R, t])  # projection-like matrix (simplified)
```

---

## Compact choices and updates

### Conditional expression: `a if condition else b`

A conditional expression is a one-line way to choose between two values. It is helpful when you need a simple decision right where a value is being assigned. In `templering_sfm.py`, it is used to pick configuration sources and defaults, for example selecting a calibration matrix from the dataset unless a YAML file is provided. Generic use is to keep small “either/or” decisions close to the variable they define. In this code, it reduces boilerplate and makes the “fallback” rule clear: use the provided input when present, otherwise rely on built-in or dataset defaults.

```python
use_yaml = False
K = "from_yaml" if use_yaml else "from_dataset"
```

### Augmented assignment like `+=` for “update in place”

Augmented assignment updates a value using its current value, such as `x += 1`. This is common in numeric code and loops because it is concise and shows “this variable is being adjusted.” In `templering_sfm.py`, one example is `b2[j] += eps` inside a numerical-derivative routine: a copy of a parameter vector is nudged by a small amount to estimate a slope. Generic use is to express small incremental changes without repeating the variable name. In this code, it makes the “perturb one coordinate” step visually obvious, which is important for correctness when building Jacobians.

```python
eps = 1e-6
b2 = [0.0, 0.0, 0.0]
j = 1
b2[j] += eps
```

---

## Taking parts of a sequence

### Slicing with `[:3]`, `[3:]`, and similar forms

Slicing selects a portion of a list, string, or array using start and end positions. It is a simple way to split data into meaningful parts. In `templering_sfm.py`, slicing is used when a 6-number pose update is split into rotation and translation components, such as `b2[:3]` for the first three values and `b2[3:]` for the remaining values. Generic use is to avoid manual indexing and to keep “take the first chunk” or “take the rest” readable. In this code, slicing keeps optimization code clean while manipulating stacked parameter vectors.

```python
x = [10, 20, 30, 40, 50, 60]
rot = x[:3]
trans = x[3:]
```

---

## Building dictionaries and defaults

### Dictionary comprehension `{k: v for ...}`

A dictionary comprehension builds a dictionary in a compact way. It is useful when you already have items that can be turned into key–value pairs, and you want the result in one expression. In `templering_sfm.py`, this style is used to build observation maps, turning track identifiers into stored 2D measurements. Generic use is to transform data while you build the final dictionary, often applying small conversions like casting IDs to integers. In this code, it supports building the “observations by track id” structure that later steps use for triangulation and bundle adjustment.

```python
pairs = [(101, "p1"), (102, "p2")]
obs = {int(tid): val for tid, val in pairs}
```

### `defaultdict` to avoid “check if key exists” boilerplate

A `defaultdict` is a dictionary that automatically creates a default value the first time a missing key is accessed. This is handy when each key should map to a growing list. In `templering_sfm.py`, `track_hist` is a `defaultdict(list)`, which allows the code to append to `track_hist[tid]` even when `tid` is seen for the first time. Generic use is to simplify “grouping” logic. Here, it keeps tracking history code straightforward: store per-track sequences without writing repeated “if key not in dict: create list” checks.

```python
from collections import defaultdict

track_hist = defaultdict(list)
tid = 7
track_hist[tid].append((0, (123.4, 56.7)))
```

---

## Iteration helpers that keep code readable

### Generator expression `(x for x in items)` for “values on demand”

A generator expression produces values one at a time instead of building a full list immediately. This is useful when you only need to feed values into another operation, such as `max`, `sum`, or `any`. In `templering_sfm.py`, it appears in places like computing an image dimension from resized images, using a form like `max(im.size[0] for im in resized)`. Generic use is to reduce temporary objects and keep the intent focused on “compute a summary.” In this code, it supports quick aggregation steps during preprocessing without creating extra intermediate lists.

```python
widths = [320, 640, 800]
m = max(w for w in widths)
```

### `zip(...)` for walking multiple sequences together

`zip` lets you iterate over multiple sequences in lockstep, yielding pairs (or tuples) of corresponding items. This is useful when two lists represent aligned information, such as “each keyframe has a matching optimized pose.” In `templering_sfm.py`, `zip` is used when pairing keyframes with optimized results and when building structures from two parallel sequences. Generic use is to avoid manual indexing, which is easy to get wrong. In this code, `zip` makes it clearer that two streams of data are intended to match position-by-position.

```python
keyframes = ["kf0", "kf1", "kf2"]
poses = ["p0", "p1", "p2"]

for kf, p in zip(keyframes, poses):
    pass
```

### `range(n)` for counted loops

`range(n)` produces a sequence of integers from 0 up to `n - 1`. It is commonly used when the number of repetitions matters more than the values in a list. In `templering_sfm.py`, it is used in numeric routines where a loop must run a fixed number of times, such as iterating over 6 pose parameters when computing numerical derivatives. Generic use is to express “repeat this step N times” clearly. In this code, it fits places where the loop index itself has meaning, like selecting which coordinate of a parameter vector to perturb.

```python
for j in range(6):
    pass
```

---

## Small functions created in place

### `lambda` for short “one-off” functions

A `lambda` is an unnamed function written inline. It is useful when a library function expects “a function argument,” but the logic is tiny and only used once. In `templering_sfm.py`, `lambda` appears when sorting a list of scored candidates, such as sorting by the score stored in the first element of each pair. Generic use is to keep sorting and selection readable without introducing a separate named function elsewhere. In this code, it makes the sorting rule explicit at the call site: “sort by score,” which supports loop-closure candidate selection.

```python
scored = [(10, "a"), (3, "b"), (7, "c")]
scored.sort(key=lambda x: x[0], reverse=True)
```

---

## Ignoring extra returned values cleanly

### Extended unpacking with `*_` to capture “the rest”

Extended unpacking allows you to assign some returned values to named variables while collecting any extra values into a list. The pattern `*_` is often used when you explicitly do not care about those extras. In `templering_sfm.py`, a projection function returns multiple outputs, but some call sites only need the first two, so you see a form like `uv2, z2, *_ = ...`. Generic use is to keep the assignment honest: it shows that there are more outputs, while still keeping the code focused on the outputs that matter at that moment. In this code, it helps numerical Jacobian code stay uncluttered.

```python
def project():
    return "uv", 1.23, "extra1", "extra2"

uv, z, *_ = project()
```

---

## Handling problems without crashing

### `try/except ... as e` to recover or report a clear error

A `try/except` block lets the program attempt an operation that might fail, then handle the failure in a controlled way. The `as e` part stores the error object so the program can log it or decide what to do next. In `templering_sfm.py`, `_load_config_json` uses this pattern when reading and parsing a JSON config file. If parsing fails, the program can respond with a clear message and fall back to defaults, rather than continuing with broken configuration data. Generic use is to protect boundaries like file I/O and parsing. In this code, it keeps “bad config” from turning into confusing failures later.

```python
import json

def load_config(text):
    try:
        return json.loads(text)
    except Exception as e:
        # simplified: return defaults after recording the error
        return {}
```

### `try/finally` to guarantee cleanup

A `finally` block runs no matter what happens in the `try` block. This is used for cleanup steps that must happen even when an error occurs. In `templering_sfm.py`, `load_K_yaml` uses `try/finally` to ensure an OpenCV file handle is released, so resources are not left open. Generic use is to protect cleanup work like closing files, releasing locks, or freeing handles. In this code, it makes the calibration loading routine safer: whether reading succeeds or fails, the OpenCV `FileStorage` is released, keeping the rest of the program stable.

```python
def read_something(resource):
    try:
        return "data"
    finally:
        resource.close()
```

---

## Path joining that reads naturally

### `PathA / "child"` for building file paths

When using `pathlib.Path`, the `/` operator can be used to join paths in a readable way. Instead of manually adding slashes or worrying about platform differences, you combine a base path and a child name with a single operator. In `templering_sfm.py`, this appears in helpers like `_default_config_path`, which builds a default location for a JSON configuration file. Generic use is to keep file path construction clear and less error-prone. In this code, it supports locating resources relative to the repository structure, and it keeps path logic consistent across operating systems.

```python
from pathlib import Path

root = Path("/project")
cfg_path = root / "config.json"
```
