# cache-probes

Three timing probes of a CPU core's store path, written to run as they are on Linux, macOS and Windows (x86-64 and
AArch64; any other ISA through GCC or Clang), and a workflow that runs them on GitHub's hosted runners. Each probe
answers from timing alone, compares layouts that run the same instructions over the same bytes, pairs and
interleaves its samples within one process, and decides with a Mann-Whitney test at a stated false-positive rate.
Each file's head comment is its full design.

| file | question |
|---|---|
| `store_probe.c` | Does a cache that every store passes through (a write-coalescing cache in front of a write-through first level, say) stall k streams written in lockstep once more of their lines share one of its sets than it has ways? If so, its line, sets and ways. Strides and stream counts are bounded so that the first-level data cache itself is never overflowed. |
| `streamer_probe.c` | How many pages of concurrent streams the second level's stream prefetcher follows before lines arrive on demand, for stores alone and for loads paired with stores, and what a demand line costs against a prefetched one. |
| `crowd_probe.c` | What one visit of a pass pays when it stores more whole lines into one first-level data cache set than the set has ways: the least crowding that costs (the knee) and the extra time per crowded line, in nanoseconds and in instructions of a reference loop shaped like an FFT pass (vector loads, a shuffle, a multiply, a fused multiply-add, an add, a subtract, vector stores). |
| `probe_os.h` | What the probes ask of the OS and the compiler: a monotonic clock, one logical CPU, memory (with or without transparent huge pages on Linux), every cache record the OS reports, 16-byte vector loads and stores. |
| `loop_insns.py` | Counts the instructions one iteration of `crowd_probe`'s reference loop retires, from a disassembly of the build (GNU objdump, llvm-objdump, otool or dumpbin), so the instruction price is never an assumed count. |

## Running

    cc -O2 -std=gnu11 -o store_probe store_probe.c -lm        # likewise streamer_probe, crowd_probe
    ./store_probe decide
    ./streamer_probe decide 10 4 100
    objdump -d --no-show-raw-insn crowd_probe > crowd.dis
    ./crowd_probe run "$(python3 loop_insns.py crowd.dis reference_loop "$(./crowd_probe fmas)" | tail -n 1)"

Under MSVC: `cl /O2 /std:c11 crowd_probe.c`, then `dumpbin /disasm:nobytes crowd_probe.obj`.

Each probe pins itself to the last logical CPU the process may use (Linux, Windows). macOS has no affinity API; the
probes ask for the highest quality of service, which the scheduler serves on performance cores first, and their
answers are for whichever core ran them.

The geometry each probe uses is the OS's: Linux sysfs, macOS `sysctl hw.*` (sizes and the line; macOS reports no
associativity), Windows `GetLogicalProcessorInformationEx`. Where the OS reports no ways, `store_probe` bounds its
streams by the least ways of a cache indexed within the page (size / page) and `crowd_probe` takes the page as the
way span; both say so in their output.

## The workflow

`.github/workflows/probes.yml` (manual dispatch) runs `probe-job.yml` on ubuntu-24.04, ubuntu-24.04-arm,
windows-latest, windows-11-arm, macos-latest and macos-15-intel: the CPU and cache facts the OS reports, the build,
the reference loop's count, `store_probe decide` in 10 fresh processes, `streamer_probe decide 10 4 100` in 5,
`crowd_probe run` in 3, all raw output uploaded as artifacts. A RISC-V job (RISE and Cloud-V runners) runs once their
GitHub Apps are installed on the repository and the repository variable `RISCV_RUNNERS` is `true`.

Hosted runners are shared virtual machines and their CPU model is not promised: read each answer with the CPU the
job reports and with the agreement across its fresh processes.
