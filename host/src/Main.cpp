// stemnotes-host: stdin/stdout JSON-line server around the native audio engine.
#include <juce_gui_basics/juce_gui_basics.h>
#include "Engine.h"
#include "Commands.h"
#include "StdioGuard.h"
#include <cstdio>
#include <iostream>
#include <string>

using namespace juce;

static FILE* proto = stdout;   // protocol channel (the original stdout)
static CriticalSection outLock;
static void writeLine(const var& v) {
    const ScopedLock sl(outLock);
    String s = JSON::toString(v, true);
    std::fputs(s.toRawUTF8(), proto); std::fputc('\n', proto); std::fflush(proto);
}
static void sendEvent(const String& name, const var& payload) {
    auto* o = new DynamicObject();
    o->setProperty("event", name);
    o->setProperty("data", payload);
    writeLine(var(o));
}

class StdinThread : public Thread {
public:
    StdinThread(std::function<void(String)> fn) : Thread("stdin"), onLine(std::move(fn)) {}
    void run() override {
        std::string line;
        while (!threadShouldExit() && std::getline(std::cin, line)) {
            if (!line.empty()) onLine(String::fromUTF8(line.c_str()));
        }
        onLine({});   // EOF -> quit
    }
    std::function<void(String)> onLine;
};

int main(int argc, char** argv) {
    proto = sn::takeProtocolStdout();
    ScopedJuceInitialiser_GUI juceInit;
    std::fprintf(stderr, "stemnotes-host %s starting\n", STEMNOTES_HOST_VERSION);
    sn::Engine engine;
    sn::Commands commands(engine);
    engine.onEvent = [](const String& name, const var& payload) { sendEvent(name, payload); };
    bool quit = false;
    commands.onQuit = [&quit]() { quit = true; };

    StdinThread reader([&](String line) {
        MessageManager::callAsync([&, line]() {
            if (line.isEmpty()) { quit = true; return; }
            var cmd = JSON::parse(line);
            if (!cmd.isObject()) { std::fprintf(stderr, "bad json: %s\n", line.toRawUTF8()); return; }
            const var id = cmd["id"];
            commands.handle(cmd, [id](bool ok, const var& result) {
                auto* o = new DynamicObject();
                o->setProperty("id", id);
                o->setProperty("ok", ok);
                if (ok) o->setProperty("result", result); else o->setProperty("error", result);
                writeLine(var(o));
            });
        });
    });
    reader.startThread();

    // transport position at ~30 Hz
    struct PosTimer : public Timer {
        sn::Engine& e; PosTimer(sn::Engine& en) : e(en) {}
        void timerCallback() override {
            auto* o = new DynamicObject();
            o->setProperty("sec", e.position()); o->setProperty("playing", e.playing()); o->setProperty("end", e.endSec());
            o->setProperty("editor_open", e.editorOpen());
            sendEvent("position", var(o));
        }
    } posTimer(engine);
    posTimer.startTimer(33);

    auto* mm = MessageManager::getInstance();
    while (!quit) mm->runDispatchLoopUntil(50);
    posTimer.stopTimer();
    reader.signalThreadShouldExit();
    engine.shutdown();
    std::fprintf(stderr, "stemnotes-host exiting\n");
    std::fflush(proto); std::fflush(stdout); std::fflush(stderr);
    // the stdin reader may be blocked in getline(); do not wait for it
    std::_Exit(0);
}
