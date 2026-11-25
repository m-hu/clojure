// Probe (throwaway, #271): advise the executable's own mappings at load,
// before main, to see what fault-around costs in peak RSS.
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
__attribute__((constructor)) static void advise_self(void) {
    const char *mode = getenv("ADVISE");
    if (!mode) return;
    int adv = strcmp(mode, "random") == 0 ? MADV_RANDOM : strcmp(mode, "nohuge") == 0 ? MADV_NOHUGEPAGE : -1;
    if (adv < 0) return;
    char exe[4096]; ssize_t n = readlink("/proc/self/exe", exe, sizeof exe - 1); if (n <= 0) return; exe[n] = 0;
    FILE *f = fopen("/proc/self/maps", "r"); if (!f) return;
    char line[8192];
    while (fgets(line, sizeof line, f)) {
        unsigned long lo, hi; char perm[8]; char path[4096] = "";
        if (sscanf(line, "%lx-%lx %7s %*s %*s %*s %4095s", &lo, &hi, perm, path) < 3) continue;
        int all = getenv("ADVISE_ALL") != NULL;
        if ((all && path[0] == '/') || strcmp(path, exe) == 0) madvise((void *)lo, hi - lo, adv);
    }
    fclose(f);
}
