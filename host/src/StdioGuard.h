#pragma once
#include <cstdio>

namespace sn {
// Moves the original stdout to a private FILE* for the JSON protocol and points stdout (C runtime and, on Windows,
// the process std handle) at stderr. Plugins print debug text to stdout; after this it lands in the host log instead
// of corrupting protocol lines. Call before any plugin is loaded. Returns stdout itself if duplication fails.
FILE* takeProtocolStdout();
}
