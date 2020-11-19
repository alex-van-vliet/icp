# GPGPU

CPU & GPU implementations of Iterative Closest Point using a Vantage-Point Tree.

## Examples

### Horse Transformation 1

![Horse Tr 1](https://raw.githubusercontent.com/alex-van-vliet/icp/master/data/horse/images/horse_tr1.gif)

### Horse Transformation 2

![Horse Tr 2](https://raw.githubusercontent.com/alex-van-vliet/icp/master/data/horse/images/horse_tr2.gif)

## Build

To build the project, run the following commands:
```
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```

This generates an executable at `build/bin/main`.

## Usage

To run the project, run the following command from inside the `build` directory:
```
./bin/main <path/to/reference.txt> <path/to/transformed.txt> [options]
```

Options can be any of the following:

- `--device cpu|gpu`: the device to run on [default: gpu]
- `--capacity <capacity>`: the vp tree capacity [default: 32 on cpu, 256 on gpu]
- `--iterations <iterations>`: the maximum number of iterations [default: 200]
- `--error <error>`: the error threshold [default: 1e-5]

This should output on stdout:
- the transformation matrix.

This should output on stderr:

- the parameters used,
- the progress,
- the first ten points of the reference, and their closest point of the transformed with the transformation applied.

## Tests

To build the tests, run the following commands if you haven't ran them earlier:
```
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
```
Then run:
```
make libcpu_tests
```

This generates an executable at `build/bin/libcpu_tests`. You can run this executable to test the lib cpu.

## Benchmarks

To build the benchmarks, run the following commands if you haven't ran them earlier:
```
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
```
Then run:
```
make bench
```

Note: the benchmarks only compile in RELEASE build type.

This generates an executable at `build/bin/bench`. You can run this executable to launch the benchmarks.
We recommand redirecting stderr to a file or null: `./bin/bench 2>/dev/null`.
