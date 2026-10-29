// Kept apart from JUCE so <windows.h> does not clash with JUCE names.
#include "StdioGuard.h"
#ifdef _WIN32
 #define NOMINMAX
 #define WIN32_LEAN_AND_MEAN
 #include <windows.h>
 #include <io.h>
#else
 #include <unistd.h>
#endif

namespace sn {

FILE* takeProtocolStdout() {
    std::fflush(stdout);
#ifdef _WIN32
    const int fd = _dup(_fileno(stdout));
    if (fd < 0) return stdout;
    _dup2(_fileno(stderr), _fileno(stdout));
    // plugins linked against their own C runtime take their stdout from this handle when they load
    SetStdHandle(STD_OUTPUT_HANDLE, GetStdHandle(STD_ERROR_HANDLE));
    FILE* f = _fdopen(fd, "wb");
#else
    const int fd = dup(STDOUT_FILENO);
    if (fd < 0) return stdout;
    dup2(STDERR_FILENO, STDOUT_FILENO);
    FILE* f = fdopen(fd, "w");
#endif
    return f ? f : stdout;
}

} // namespace sn
