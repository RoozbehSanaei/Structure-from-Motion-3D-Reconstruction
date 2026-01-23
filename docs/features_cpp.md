# C++ Language Features Illustrated (Non-basic)

This document lists C++ language features (excluding the items already present in the provided reference list), with one feature per paragraph and related concepts grouped into sections.

## Pointer Safety

### `nullptr` literal
In a C++ version of this pipeline, `nullptr` is used to represent “no object here” in a way that cannot be confused with a normal number, such as when an image failed to load or a component was not created; you typically see it in steps like `Image* img = nullptr;` and later checks like `img == nullptr`.

## Type-Conscious Conversions

### `static_cast<...>` for “normal” conversions
When the code needs to convert between related numeric types without surprises (for example, turning an integer count into a floating-point value for a calculation), it uses `static_cast<double>(count)` so the conversion is explicit and readable, rather than relying on quiet, hard-to-notice automatic conversions.

### `dynamic_cast<...>` for safe downcasting
If the program stores a generic “base” object (for example, a generic estimator interface) and sometimes needs a more specific implementation, `dynamic_cast<Derived*>(base)` is used to safely ask “is this really a Derived,” and the result is either a usable pointer or a null-like value, which avoids calling the wrong methods on the wrong kind of object.

### `reinterpret_cast<...>` for low-level reinterpretation
When the program must treat a value as raw bits for interoperability or diagnostics (for example, turning an address into an integer for logging), `reinterpret_cast<std::uintptr_t>(p)` performs that change of view explicitly, making it clear that this is a low-level operation that should be used sparingly.

## Controlled Failure Handling

### `try` / `catch` blocks
When calling third-party code that may report failures by throwing exceptions (for example, a computer-vision library failing to parse or compute something), the code wraps the risky region in `try { ... }` and then handles the failure in `catch (cv::Exception e) { ... }`, keeping the program from crashing and allowing it to print a clear message or skip the bad item.

### Catch-all handler `catch (...)`
For “anything went wrong, regardless of the error type” situations—especially at a top-level boundary—the code can use `catch (...) { ... }` so one unexpected failure does not terminate the whole batch run, which is helpful when processing many frames or many images.

### Rethrow with `throw;`
Inside layered error handling, `throw;` is used when a lower-level block wants to do a small amount of cleanup or add context, yet still wants the original error to continue upward unchanged, so higher-level code can decide whether to abort or keep going.

## Clear States and Intent

### Scoped enumerations with `enum class`
When the program wants a small set of named states that cannot be mixed with ordinary numbers (for example, the step of the pipeline currently running), it defines a scoped enum like `enum class Stage { Detect, Match, Pose };` and later uses it with names such as `Stage::Pose`, which makes state values self-explanatory and hard to misuse.

## Extensible Design

### Virtual functions via `virtual`
If the code defines a “plug-in point” where different implementations can be swapped (for example, different feature detectors or matchers), it declares an interface function like `virtual bool estimatePose(...);` so the program can call `estimatePose(...)` and automatically run the correct implementation without needing to know the exact concrete type at that call site.

### Override checking with `override`
When a derived component is meant to replace a base function (for example, a specialized matcher replacing a generic matcher), the derived declaration adds `override`, as in `bool estimatePose(...) override;`, so the compiler verifies it truly matches an existing base function and prevents subtle bugs caused by mismatched names or parameters.

## Object Construction Rules

### Defaulted special members with `= default`
When the code wants the standard, “do the usual thing” behavior for construction, it uses `Tracker() = default;`, which keeps the class definition short and avoids writing boilerplate that can accidentally change behavior, while still clearly stating that the default behavior is intended.

### Deleted operations with `= delete`
When the program wants to forbid a particular action because it would lead to confusing or unsafe states, it can delete it explicitly, such as `Tracker() = delete;`, which prevents creation in an invalid way at compile time rather than letting it fail later during execution.

## Expressive Operations

### Operator overloading with `operator*`
In math-heavy code, it is common to express calculations in a readable way by defining operations on custom types, such as `Matrix operator*(Matrix a, Matrix b);`, which lets the rest of the code read more like the underlying math and reduces repetitive helper function calls.

### Friendship with `friend`
When a helper needs access to internal details for a focused purpose (for example, a debug print that needs to see private fields without adding many public “getter” functions), the code can grant access explicitly using a declaration like `friend void logState(Tracker);`, keeping the public surface smaller and the intent very targeted.

### User-defined literals with `operator""`
To make certain numeric values self-describing (for example, representing a pixel unit or a degree-like value in a readable way), the code can define a literal operator such as `double operator"" _px(long double v);` and then write values like `12.5_px`, which reduces confusion about what a plain number is supposed to represent.

## Per-Thread Storage

### `thread_local` variables
When the code runs work in parallel and needs each thread to have its own independent cached data (for example, a scratch buffer used during matching), it can declare `thread_local int cache;`, which prevents threads from accidentally stepping on each other’s temporary values.

## Concurrency Primitives

### `std::thread` for running work concurrently
If the program processes many images or pairs and wants to do some of that at the same time, it can launch work with `std::thread t(worker);` and then synchronize completion with `t.join();`, which allows the pipeline to use available CPU cores while keeping the program’s end state predictable.

### `std::mutex` for protecting shared data
When multiple parts of the program might touch the same shared object (for example, appending to a shared results list), `std::mutex m;` provides a “one-at-a-time” gate so the code can lock before updating and unlock after, which prevents corrupted results caused by simultaneous writes.

### `std::condition_variable` for coordinated waiting
When one part of the program must wait until another part finishes a prerequisite step (for example, waiting until a batch of frames has been loaded), `std::condition_variable cv;` supports signaling like `cv.notify_one();` and waiting like `cv.wait(lk);`, which avoids wasteful spinning and keeps coordination explicit.

## Practical File Paths

### `std::filesystem::path` for robust path handling
When working with dataset files and output locations, `std::filesystem::path p("templeR_par.txt");` lets the code build, join, and convert paths in a consistent way (for example, producing a string with `p.string();`), which reduces errors that come from manual string concatenation and makes path logic easier to maintain across operating systems.
