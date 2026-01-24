## Scoped choices with `enum class`

A scoped enumeration is a small, fixed menu of named choices. It helps code stay readable because you work with words (like `MESH`) instead of “magic numbers” (like `3`). It also keeps the names “scoped” to the enum type, which reduces accidental name clashes. In this code, `ExportGeometry` represents what kind of geometry output to produce. Using named options makes later logic easier to follow, because the program can store and check a clear choice rather than relying on string comparisons everywhere.

```cpp
enum class ExportGeometry {
  NONE=0, POINTCLOUD=1, MESH=2, BOTH=3
};
```

## Flexible path arguments with `std::initializer_list`

`std::initializer_list` lets a function accept a brace-written list like `{ "a", "b", "c" }` without the caller having to create a separate container first. Think of it as “handing a short list of steps” to a function. In this program, the JSON lookup helpers accept paths expressed as a short sequence of keys. That makes call sites concise and readable, because the caller can pass a path in-place rather than building an array or vector just to pass a few items.

```cpp
static const minijson::Value* jpick(const minijson::Value& root,
                                   std::initializer_list<const char*> a,
                                   std::initializer_list<const char*> b)
{
```

## Small helpers with `inline`

`inline` marks a function as a lightweight helper that is intended to be cheap to call. A practical way to think about it is: this helper is used a lot, so the compiler is given more freedom to make the call overhead minimal. In this code, pixel sampling is a tiny operation done repeatedly inside tracking and image processing. Declaring it `inline` fits that usage pattern: it communicates “this is a small building block” that can be used many times without cluttering performance-critical loops.

```cpp
static inline double sample_bilinear(const GrayImage& im, double x, double y){
```

## Type-level utility via a `static` member function

A `static` member function belongs to the type itself, rather than to one specific object. You call it with `TypeName::FunctionName()` and you do not need an instance first. In this program, the pose structure provides an `Identity()` function that creates a standard “do-nothing” pose. This is useful as a clear starting value when building up camera motion and transformations, because the caller can request a known default pose in one obvious line.

```cpp
struct PoseCW {
  Mat33 Rcw;
  Vec3 tcw;
  static PoseCW Identity(){ return {Mat33::I(), {0,0,0}}; }
};
```

## “Local-only” data records with a struct declared inside a function

C++ allows you to define a `struct` inside a function when the type is only meaningful there. This keeps the type from leaking into the rest of the file and reduces the mental load for readers: the record exists only where it is used. In this program, a small `Cand` record is used to hold candidate points during feature selection. By defining it locally, the code communicates that “Cand is just a temporary shape for this one procedure,” not a reusable global type.

```cpp
struct Cand { int x,y; double s; };
```

## Read-only promise with a `const` member function

When a member function ends with `const`, it promises that calling it will not change the object’s stored data. This matters in two ways: it makes intent clear to humans, and it lets the compiler allow calls on “read-only” objects. In this code, `MapState::has` is a check: it answers whether a track id is already known. Because it should only check and not modify, the `const` marker matches the real purpose of the function.

```cpp
struct MapState {
  std::unordered_map<int,int> tid2pid;
  std::unordered_map<int, MapPoint> pts;
  bool has(int tid) const { return tid2pid.find(tid) != tid2pid.end(); }
  int add(int tid, Vec3 Xw){
```

## Passing by reference with `&` parameters

A reference parameter (written with `&`) lets a function work directly with the caller’s object instead of making a full copy. For large objects, copying can be slow and unnecessary. A `const &` reference is “read-only”: the function can look but cannot change it. In this program, the bundle adjustment routine takes large structures like matrices and configuration objects by reference. That matches how the function is used: it needs access to shared state and should avoid making duplicates of big data.

```cpp
static void bundle_adjust_window(const Mat33& K,
                                 std::vector<Keyframe>& kfs,
                                 MapState& map,
                                 const BAConfig& cfg)
{
```

## Safe “start from zero” with value-initialization `{}`

Writing `{}` after a variable tells C++ to initialize it to a clean default state. For numeric storage, that typically means zeroed values. This is useful because it avoids “leftover garbage” data that can happen when a variable is created without initialization. In this program, matrices used for calculations are created with `{}` before individual entries are filled in. That reduces the risk of accidental use of uninitialized elements during intermediate steps of math-heavy routines.

```cpp
Mat33 inv{};
inv(0,0) =  (K(1,1)*K(2,2)-K(1,2)*K(2,1))/d;
```

## Fixed-size storage with `std::array`

`std::array` is a fixed-size container: its length is part of the type, so it always has the same number of slots. This is helpful when the amount of data is known up front, like a fixed-length record read from a dataset. In this code, a 21-value block is read into an array, then different parts of that array are used to fill camera calibration and pose fields. Using `std::array` makes the “exactly 21 values” expectation explicit.

```cpp
std::array<double,21> v{};
```

## Grow-as-needed lists with `std::vector`

`std::vector` is a growable list. You can start empty and append items as you discover them, which is common when reading files or building results step-by-step. In this program, records are accumulated into a vector after reading the dataset. The code also calls `reserve(...)` to pre-allocate enough space ahead of time, which reduces repeated resizing as elements are appended. This matches the workflow: read N items, store N items, then return the complete list.

```cpp
std::vector<MBRecord> recs;
recs.reserve((size_t)n);
// ...
recs.push_back(r);
```

## Fast key lookup with `std::unordered_map`

An `std::unordered_map` stores pairs of (key → value) so you can quickly find a value when you know its key. The “unordered” part means it is organized for speed rather than for sorted order. In this code, `kf2local` maps a keyframe id to a local index in a sliding window. That is exactly the kind of lookup where a hash map helps: later steps can jump from an id to its index without scanning a whole list.

```cpp
std::unordered_map<int,int> kf2local;
kf2local.reserve((size_t)W);
for(int li=0; li<W; ++li) kf2local.emplace(kfs[w0+li].kf_id, li);
```

## Membership tracking with `std::unordered_set`

`std::unordered_set` stores “just keys” and is used when you only care whether something is present. It is a natural fit for “have we used this already?” checks. In this program, the mesh export path uses a grid-based subsampling step. It tracks which grid cells have already been taken so the output points do not cluster into near-duplicates. The set supports fast membership tests (`find`) and fast insertion (`insert`) as new cells are accepted.

```cpp
std::unordered_set<CellKey, CellHash> used;
used.reserve((size_t)max_points*2);
// ...
if(used.find(ck) != used.end()) continue;
used.insert(ck);
```

## Returning two related results with `std::pair`

Sometimes a function naturally produces two outputs that should stay together. `std::pair` is a simple way to return exactly two items as one unit. In this code, `inv_wc()` computes an inverse pose representation and returns both the rotation and the translation together. This keeps the return value compact and avoids creating a custom struct just for two fields, while still making it clear that the two results belong to the same computed transform.

```cpp
std::pair<Mat33, Vec3> inv_wc() const {
  const Mat33 Rwc = sfm::transpose(Rcw);
  const Vec3 twc = -(Rwc * tcw);
  return {Rwc, twc};
}
```

## Repeatable randomness with `std::mt19937`

`std::mt19937` is a pseudo-random number generator. It produces a sequence of “random-looking” values. When you give it a fixed seed (like `12345`), the sequence becomes repeatable, which is valuable for debugging: the same run produces the same sampling decisions. In this program, random sampling is used in a robust estimation loop where small subsets of correspondences are tried. Seeding the generator makes outcomes stable across runs while still providing variability inside the algorithm.

```cpp
std::mt19937 rng(12345);
```

## Picking a random index range with `std::uniform_int_distribution`

A uniform integer distribution chooses integers within a specified inclusive range, with each integer equally likely. This is handy when you want to select random indices from a list. In this code, the distribution is set up to select indices that match the size of the input point list. Inside the loop, it fills an array of 8 indices by drawing from the distribution repeatedly. That directly supports the “pick a small random sample” step used in robust estimation.

```cpp
std::uniform_int_distribution<int> uni(0, (int)pi.size()-1);
// ...
for(int k=0;k<8;k++) idx8[k] = uni(rng);
```

## Randomizing order with `std::shuffle`

`std::shuffle` rearranges the elements of a container into a random order. A practical reason to shuffle is to avoid bias that comes from the original ordering (for example, spatial ordering or file ordering). In this program, candidate points are shuffled before selecting up to a maximum number of points. That means early elements are not always favored, and the selection process can be closer to a fair sample of the candidates.

```cpp
std::shuffle(cands.begin(), cands.end(), rng);
```

## Compact “choose A or B” logic with the ternary operator `?:`

The ternary operator is a short form of an if/else that chooses between two expressions. It is useful when the decision is small and the code benefits from staying on one line. In this program, it is used to decide which condition to evaluate based on orientation. The result is still a clear “two-way choice,” just written more compactly. Readers can interpret it as: if the first check is true, use the middle expression; otherwise use the last expression.

```cpp
return (o > 0.0) ? (det > 1e-12) : (det < -1e-12);
```

## Reading values from a stream with `>>`

The `>>` operator pulls values out of a stream (like a file) and places them into variables. It is a simple way to parse whitespace-separated text: each extraction reads the next token and converts it to the variable’s type. In this code, `>>` is used to read an integer count, then repeatedly read an image name and a fixed set of numeric parameters. That matches the format of the dataset files: structured rows of numbers that are easy to ingest step-by-step.

```cpp
int n=0;
f >> n;
// ...
f >> r.img;
for(int k=0;k<21;k++) f >> v[k];
```

## Writing output with `<<`

The `<<` operator pushes values into an output stream. It supports chaining, so the program can build one readable line from multiple parts without manual string concatenation. In this code, progress messages are printed as the run proceeds, including the current frame index and counts of internal objects. This makes the program’s behavior visible while it runs, which is useful for long computations where you want reassurance that work is progressing normally.

```cpp
std::cout << "frame " << (fi+1) << "/" << frames << " | keyframes=" << kfs.size()
          << " | map_points=" << map.pts.size() << "\n";
```

## Signaling failure with `throw std::runtime_error`

Throwing a `std::runtime_error` is a way to stop normal execution when the program cannot continue safely, while also attaching a human-readable message explaining why. In this code, file opening is a critical prerequisite. If a required file cannot be opened, the helper throws an error immediately instead of continuing with missing data. This makes failures loud and specific, which is usually better than producing silent incorrect results later.

```cpp
if(!f) throw std::runtime_error("Failed to open: " + p.string());
```
