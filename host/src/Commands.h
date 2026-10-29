#pragma once
#include "Engine.h"

namespace sn {

// JSON-line protocol: {"id":N,"op":"...", ...} -> {"id":N,"ok":true,"result":...} ; events: {"event":"...", ...}
class Commands {
public:
    explicit Commands(Engine& e) : engine(e) {}
    // Runs on the message thread. `reply` may be called later (async operations).
    void handle(const juce::var& cmd, std::function<void(bool ok, const juce::var& resultOrError)> reply);
    std::function<void()> onQuit;
private:
    Engine& engine;
};

} // namespace sn
