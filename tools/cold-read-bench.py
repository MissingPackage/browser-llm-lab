import os, time, random, json
PATH = os.path.expanduser("~/.cache/blab-cold-bench.bin")
GIB = 1024**3
SIZE = 6 * GIB
TILE = 256 * 1024**2
EXPERT = 5325512  # byte/expert misurati (residency-sim)

# 1. genera file incomprimibile (tile urandom)
if not (os.path.exists(PATH) and os.path.getsize(PATH) == SIZE):
    t0 = time.time()
    with open("/dev/urandom", "rb") as r, open(PATH, "wb") as w:
        tile = r.read(TILE)
        for _ in range(SIZE // TILE):
            w.write(tile if _ == 0 else r.read(TILE))
        w.flush(); os.fsync(w.fileno())
    print(f"write: {SIZE/GIB:.0f} GiB in {time.time()-t0:.1f}s")

fd = os.open(PATH, os.O_RDONLY)

def drop():
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    time.sleep(1)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)

def bench_seq(label, nbytes, block=1024*1024, offset=0):
    os.lseek(fd, offset, os.SEEK_SET)
    t0 = time.perf_counter(); got = 0
    while got < nbytes:
        b = os.read(fd, block)
        if not b: break
        got += len(b)
    dt = time.perf_counter() - t0
    print(f"{label}: {got/GIB:.2f} GiB in {dt*1000:.0f} ms -> {got/dt/1e9:.2f} GB/s")
    return got/dt/1e9

def bench_rand_expert(label, n=192):
    rng = random.Random(42)
    offs = [rng.randrange(0, (SIZE - EXPERT)//4096) * 4096 for _ in range(n)]
    t0 = time.perf_counter(); got = 0
    lat = []
    for o in offs:
        t1 = time.perf_counter()
        os.lseek(fd, o, os.SEEK_SET)
        got += len(os.read(fd, EXPERT))
        lat.append((time.perf_counter()-t1)*1000)
    dt = time.perf_counter() - t0
    lat.sort()
    print(f"{label}: {n} expert da {EXPERT/1e6:.2f} MB in {dt*1000:.0f} ms -> {got/dt/1e9:.2f} GB/s | "
          f"lat/expert p50 {lat[n//2]:.2f} ms p95 {lat[int(n*0.95)]:.2f} ms")
    return got/dt/1e9

drop()
seq_cold = bench_seq("seq COLD 1MiB", 4*GIB)
seq_warm = bench_seq("seq WARM re-read", 1*GIB, offset=3*GIB)  # sanity: appena letto
drop()
rand_cold = bench_rand_expert("rand-expert COLD")
rand_warm = bench_rand_expert("rand-expert WARM (stessi offset)")
os.close(fd)
print(json.dumps({"seqColdGBps": round(seq_cold,2), "seqWarmGBps": round(seq_warm,2),
                  "randExpertColdGBps": round(rand_cold,2), "randExpertWarmGBps": round(rand_warm,2)}))
