#include "Commands.h"
using namespace juce;
namespace sn {

static DynamicObject* obj() { return new DynamicObject(); }

void Commands::handle(const var& cmd, std::function<void(bool, const var&)> reply) {
    const String op = cmd["op"].toString();
    try {
        if (op == "ping") { reply(true, "pong"); return; }
        if (op == "version") { reply(true, STEMNOTES_HOST_VERSION); return; }
        if (op == "start") {
            String err;
            String dev = engine.start(cmd["device"].toString(), (int) cmd["block"], err);
            auto* o = obj();
            o->setProperty("available", err.isEmpty());
            o->setProperty("error", err.isEmpty() ? var() : var(err));
            o->setProperty("sample_rate", engine.sampleRate());
            o->setProperty("block", engine.blockSize());
            o->setProperty("device", dev);
            reply(true, var(o)); return;
        }
        if (op == "stop_stream") { engine.stopDevice(); reply(true, true); return; }
        if (op == "devices") { reply(true, engine.listDevices()); return; }
        if (op == "load_soundfont") {
            String err; bool ok = engine.loadSoundFont(File(cmd["path"].toString()), err);
            auto* o = obj(); o->setProperty("error", ok ? var() : var(err)); reply(true, var(o)); return;
        }
        if (op == "master") { engine.setMaster((float) (double) cmd["value"]); reply(true, true); return; }
        if (op == "set_project" || op == "sync") {
            File dir(cmd["directory"].toString());
            if (op == "set_project") { engine.stop(); engine.seek(0.0); }
            engine.applyProject(cmd["data"], dir, [this, reply](String err) {
                if (err.isNotEmpty()) { reply(false, err); return; }
                reply(true, engine.vstInfo());
            });
            return;
        }
        if (op == "play") { engine.play(); reply(true, true); return; }
        if (op == "stop") { engine.stop(); reply(true, true); return; }
        if (op == "seek") { engine.seek((double) cmd["sec"]); reply(true, true); return; }
        if (op == "set_loop") { engine.setLoop((bool) cmd["enabled"], (double) cmd["start"], (double) cmd["end"]); reply(true, true); return; }
        if (op == "preview") { engine.preview(cmd["track_id"].toString(), (int) cmd["pitch"], cmd.hasProperty("velocity") ? (int) cmd["velocity"] : 100, cmd.hasProperty("duration") ? (double) cmd["duration"] : 0.4); reply(true, true); return; }
        if (op == "preview_off") { engine.previewOff(cmd["track_id"].toString(), (int) cmd["pitch"]); reply(true, true); return; }
        if (op == "vst_names") { String err; auto names = engine.pluginNames(cmd["path"].toString(), err); Array<var> a; for (auto& n : names) a.add(n); reply(true, a); return; }
        if (op == "replace_vst") {
            const String tid = cmd["track_id"].toString(), path = cmd["path"].toString(), name = cmd["plugin_name"].toString();
            engine.loadPlugin(tid, path, name, [this, reply, tid, path, name](bool ok, String err) {
                auto* o = obj(); o->setProperty("ok", ok);
                if (ok) {
                    auto info = engine.vstInfo();
                    o->setProperty("name", info["vst_tracks"][Identifier(tid)]);
                    auto* inst = obj(); inst->setProperty("type", "vst3"); inst->setProperty("path", path);
                    inst->setProperty("plugin_name", name.isEmpty() ? var() : var(name)); inst->setProperty("state", var());
                    inst->setProperty("name", info["vst_tracks"][Identifier(tid)]);
                    o->setProperty("instrument", var(inst));
                } else o->setProperty("error", err);
                reply(true, var(o));
            });
            return;
        }
        if (op == "vst_state") { String s = engine.pluginState(cmd["track_id"].toString()); reply(true, s.isEmpty() ? var() : var(s)); return; }
        if (op == "show_editor") {
            const String tid = cmd["track_id"].toString();
            String err;
            bool ok = engine.showEditor(tid, err, [this, tid]() {
                auto* o = obj(); o->setProperty("track_id", tid); o->setProperty("ok", true);
                String st = engine.pluginState(tid); o->setProperty("state", st.isEmpty() ? var() : var(st));
                if (engine.onEvent) engine.onEvent("editor_closed", var(o));
            });
            auto* o = obj(); o->setProperty("ok", ok); if (!ok) o->setProperty("error", err);
            reply(true, var(o)); return;
        }
        if (op == "close_editor") { int n = engine.closeEditors(); auto* o = obj(); o->setProperty("ok", true); o->setProperty("windows", n); reply(true, var(o)); return; }
        if (op == "editor_state") { auto* o = obj(); o->setProperty("open", engine.editorOpen()); reply(true, var(o)); return; }
        if (op == "test_vst") {
            const String tid = cmd["track_id"].toString();
            auto info = engine.vstInfo();
            if (!info["vst_tracks"].hasProperty(Identifier(tid))) { auto* o = obj(); o->setProperty("ok", false); o->setProperty("error", String(CharPointer_UTF8("на дорожке нет загруженного VST3"))); reply(true, var(o)); return; }
            engine.startTest(tid);
            const String name = info["vst_tracks"][Identifier(tid)].toString();
            Timer::callAfterDelay(1300, [this, reply, tid, name]() {
                float pk = engine.testResult(tid);
                auto* o = obj(); o->setProperty("ok", true); o->setProperty("name", name); o->setProperty("peak", (double) pk);
                o->setProperty("samples", (int) (1.3 * engine.sampleRate())); o->setProperty("seconds", 1.3); o->setProperty("params", 0);
                reply(true, var(o));
            });
            return;
        }
        if (op == "offline_render") { double pk = engine.offlineRender(cmd.hasProperty("seconds") ? (double) cmd["seconds"] : 1.0); auto* o = obj(); o->setProperty("peak", pk); o->setProperty("position", engine.position()); reply(true, var(o)); return; }
        if (op == "quit") { reply(true, true); if (onQuit) onQuit(); return; }
        reply(false, "unknown op " + op);
    } catch (std::exception& e) {
        reply(false, String("exception: ") + e.what());
    }
}

} // namespace sn
