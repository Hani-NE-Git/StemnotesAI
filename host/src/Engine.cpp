#define TSF_IMPLEMENTATION
#include "tsf.h"
#include "Engine.h"
#include <algorithm>
#include <thread>

using namespace juce;
namespace sn {

static void logLine(const String& s) { std::fprintf(stderr, "%s\n", s.toRawUTF8()); std::fflush(stderr); }

Engine::Engine() {
    formatManager.addDefaultFormats();
    audioFormats.registerBasicFormats();
    mix.setSize(2, 8192);
    sfBuf.resize(8192 * 2);
}

Engine::~Engine() { shutdown(); }

void Engine::shutdown() {
    stopDevice();
    std::vector<std::unique_ptr<MidiTrack>> old;
    { const ScopedLock sl(lock); midiTracks.swap(old); audioTracks.clear(); if (synth) { tsf_close(synth); synth = nullptr; } }
    for (auto& t : old) { t->editor.reset(); t->plugin.reset(); }
}

// ---------------------------------------------------------------- device
String Engine::start(const String& deviceName, int bufferSize, String& error) {
    stopDevice();
    auto& types = deviceManager.getAvailableDeviceTypes();
    String typeName, name = deviceName;
    if (name.contains(": ")) { typeName = name.upToFirstOccurrenceOf(": ", false, false); name = name.fromFirstOccurrenceOf(": ", false, false); }
    if (typeName.isEmpty()) {
        for (auto* t : types) if (t->getTypeName() == "Windows Audio") typeName = t->getTypeName();   // WASAPI shared
        if (typeName.isEmpty() && types.size() > 0) typeName = types[0]->getTypeName();
    }
    if (typeName.isNotEmpty()) deviceManager.setCurrentAudioDeviceType(typeName, true);
    AudioDeviceManager::AudioDeviceSetup setup;
    setup.bufferSize = bufferSize > 0 ? bufferSize : 1024;
    setup.useDefaultOutputChannels = true;
    if (name.isNotEmpty()) setup.outputDeviceName = name;
    error = deviceManager.initialise(0, 2, nullptr, true, name, &setup);
    if (error.isNotEmpty() || deviceManager.getCurrentAudioDevice() == nullptr) {
        if (error.isEmpty()) error = "no output device";
        deviceOk = false;
        logLine("device error: " + error);
        return {};
    }
    deviceManager.addAudioCallback(this);
    deviceOk = true;
    auto* dev = deviceManager.getCurrentAudioDevice();
    logLine("device started: " + dev->getTypeName() + ": " + dev->getName() + " sr=" + String(sr) + " block=" + String(block));
    return dev->getTypeName() + ": " + dev->getName();
}

void Engine::stopDevice() {
    if (deviceOk) { deviceManager.removeAudioCallback(this); deviceManager.closeAudioDevice(); }
    deviceOk = false;
}

var Engine::listDevices() {
    Array<var> out;
    for (auto* t : deviceManager.getAvailableDeviceTypes()) {
        t->scanForDevices();
        for (auto& n : t->getDeviceNames(false)) {
            auto* o = new DynamicObject();
            o->setProperty("type", t->getTypeName());
            o->setProperty("name", n);
            o->setProperty("id", t->getTypeName() + ": " + n);
            out.add(var(o));
        }
    }
    return out;
}

void Engine::audioDeviceAboutToStart(AudioIODevice* device) {
    sr = device->getCurrentSampleRate();
    block = device->getCurrentBufferSizeSamples();
    const ScopedLock sl(lock);
    if (synth) tsf_set_output(synth, TSF_STEREO_INTERLEAVED, (int) sr, 0.0f);
    for (auto& t : midiTracks) if (t->plugin) preparePlugin(*t);
    mix.setSize(2, jmax(block, 512) * 2);
    sfBuf.resize((size_t) mix.getNumSamples() * 2);
}

void Engine::audioDeviceStopped() {}

// ---------------------------------------------------------------- soundfont
bool Engine::loadSoundFont(const File& f, String& error) {
    tsf* s = tsf_load_filename(f.getFullPathName().toRawUTF8());
    if (!s) { error = "cannot load soundfont " + f.getFullPathName(); return false; }
    tsf_set_output(s, TSF_STEREO_INTERLEAVED, (int) sr, 0.0f);
    tsf_set_max_voices(s, 256);
    const ScopedLock sl(lock);
    if (synth) tsf_close(synth);
    synth = s;
    soundfont = f;
    for (auto& t : midiTracks) if (t->instType == "sf2") applySf2Program(*t);
    return true;
}

void Engine::applySf2Program(MidiTrack& t) {
    if (!synth || t.sfChannel < 0) return;
    tsf_channel_set_bank(synth, t.sfChannel, t.bank);
    if (!tsf_channel_set_presetnumber(synth, t.sfChannel, t.program, t.drums ? 1 : 0))
        tsf_channel_set_presetnumber(synth, t.sfChannel, 0, t.drums ? 1 : 0);
}

// ---------------------------------------------------------------- audio files
std::shared_ptr<AudioBuffer<float>> Engine::loadAudioFile(const File& f) {
    std::unique_ptr<AudioFormatReader> r(audioFormats.createReaderFor(f));
    if (!r) { logLine("cannot decode " + f.getFullPathName()); return nullptr; }
    const int64 len = r->lengthInSamples;
    AudioBuffer<float> tmp((int) jmax(1, (int) r->numChannels), (int) len);
    r->read(&tmp, 0, (int) len, 0, true, true);
    auto out = std::make_shared<AudioBuffer<float>>(2, (int) len);
    for (int ch = 0; ch < 2; ++ch) out->copyFrom(ch, 0, tmp, jmin(ch, tmp.getNumChannels() - 1), 0, (int) len);
    if (std::abs(r->sampleRate - sr) > 1.0) {
        const double ratio = r->sampleRate / sr;
        const int outLen = (int) (len / ratio) + 1;
        auto res = std::make_shared<AudioBuffer<float>>(2, outLen);
        for (int ch = 0; ch < 2; ++ch) {
            LagrangeInterpolator interp;
            interp.process(ratio, out->getReadPointer(ch), res->getWritePointer(ch), outLen, (int) len, 0);
        }
        out = res;
    }
    logLine("loaded " + f.getFileName() + " (" + String(out->getNumSamples() / sr, 1) + " s)");
    return out;
}

// ---------------------------------------------------------------- project
void Engine::rebuildSchedule(MidiTrack& t, const var& notes, double bpm, double offset) {
    t.notes.clear();
    if (auto* arr = notes.getArray()) {
        t.notes.reserve((size_t) arr->size());
        for (auto& n : *arr) {
            auto* a = n.getArray();
            if (!a || a->size() < 3) continue;
            NoteEvent e;
            double s = (double) (*a)[0], l = (double) (*a)[1];
            e.start = (int64_t) ((offset + s * 60.0 / bpm) * sr);
            e.end = (int64_t) ((offset + (s + l) * 60.0 / bpm) * sr);
            if (e.end <= e.start) e.end = e.start + 1;
            e.pitch = (int) (*a)[2];
            e.vel = a->size() > 3 ? (int) (*a)[3] : 100;
            if (e.end > 0) t.notes.push_back(e);
        }
    }
    std::sort(t.notes.begin(), t.notes.end(), [](const NoteEvent& a, const NoteEvent& b) { return a.start < b.start; });
    const int64_t p = pos.load();
    t.nextNote = (size_t) (std::lower_bound(t.notes.begin(), t.notes.end(), p, [](const NoteEvent& e, int64_t v) { return e.start < v; }) - t.notes.begin());
    t.active.clear();
}

void Engine::updateEnd() {
    int64_t e = (int64_t) (sr * 10);
    for (auto& a : audioTracks) if (a->buffer) e = jmax(e, (int64_t) a->buffer->getNumSamples());
    for (auto& t : midiTracks) for (auto& n : t->notes) e = jmax(e, n.end + (int64_t) (2 * sr));
    endSample = e;
}

void Engine::applyProject(const var& project, const File& dir, std::function<void(String)> done) {
    // 1) which audio files are new?
    std::vector<std::pair<String, File>> toLoad;
    if (auto* tracks = project["tracks"].getArray()) {
        for (auto& tv : *tracks) {
            if (tv["kind"].toString() != "audio") continue;
            String p = tv["path"].toString();
            if (p.isEmpty()) continue;
            File f = File::isAbsolutePath(p) ? File(p) : dir.getChildFile(p);
            const ScopedLock sl(lock);
            if (audioCache.find(f.getFullPathName()) == audioCache.end()) toLoad.push_back({ f.getFullPathName(), f });
        }
    }
    auto install = [this, project, dir, done](std::map<String, std::shared_ptr<AudioBuffer<float>>> loaded) {
        {
            const ScopedLock sl(lock);
            for (auto& kv : loaded) audioCache[kv.first] = kv.second;
        }
        const double bpm = project.hasProperty("bpm") ? (double) project["bpm"] : 120.0;
        const double offset = project.hasProperty("offset") ? (double) project["offset"] : 0.0;
        lastBpm = bpm; lastOffset = offset;
        std::vector<std::unique_ptr<AudioTrack>> newAudio;
        std::vector<std::unique_ptr<MidiTrack>> newMidi;
        std::vector<std::function<void()>> pluginLoads;
        bool solo = false;
        {
            const ScopedLock sl(lock);
            std::set<String> usedCh;
            std::map<String, MidiTrack*> old;
            for (auto& t : midiTracks) old[t->id] = t.get();
            if (auto* tracks = project["tracks"].getArray()) {
                for (auto& tv : *tracks) {
                    const String kind = tv["kind"].toString();
                    if (kind == "audio") {
                        auto a = std::make_unique<AudioTrack>();
                        a->id = tv["id"].toString(); a->mute = (bool) tv["mute"]; a->solo = (bool) tv["solo"];
                        a->volume = tv.hasProperty("volume") ? (float) (double) tv["volume"] : 0.8f;
                        String p = tv["path"].toString();
                        File f = File::isAbsolutePath(p) ? File(p) : dir.getChildFile(p);
                        a->path = f.getFullPathName();
                        auto it = audioCache.find(a->path);
                        if (it != audioCache.end()) a->buffer = it->second;
                        solo = solo || a->solo;
                        newAudio.push_back(std::move(a));
                    } else if (kind == "midi") {
                        const String id = tv["id"].toString();
                        std::unique_ptr<MidiTrack> t;
                        auto oi = old.find(id);
                        if (oi != old.end()) {
                            // reuse the existing track object (keeps the loaded plugin / editor)
                            for (auto& ot : midiTracks) if (ot && ot->id == id) { t = std::move(ot); break; }
                        }
                        if (!t) { t = std::make_unique<MidiTrack>(); t->id = id; }
                        t->name = tv["name"].toString();
                        t->mute = (bool) tv["mute"]; t->solo = (bool) tv["solo"];
                        t->volume = tv.hasProperty("volume") ? (float) (double) tv["volume"] : 0.8f;
                        solo = solo || t->solo;
                        auto inst = tv["instrument"];
                        const String type = inst.hasProperty("type") ? inst["type"].toString() : "sf2";
                        if (type == "vst3") {
                            const String path = inst["path"].toString(), name = inst["plugin_name"].toString();
                            if (t->instType != "vst3" || t->vstPath != path || t->vstName != name || (!t->plugin && !t->vstPath.isEmpty() && t->vstPath != path)) {
                                t->editor.reset();
                                t->plugin.reset();
                                t->instType = "vst3"; t->vstPath = path; t->vstName = name;
                                t->stateB64 = inst["state"].toString();
                                MidiTrack* raw = t.get();
                                pluginLoads.push_back([this, raw, path, name, id]() {
                                    loadPlugin(id, path, name, [this, id, raw](bool ok, String err) {
                                        auto* o = new DynamicObject();
                                        o->setProperty("track_id", id);
                                        o->setProperty("ok", ok);
                                        o->setProperty("name", raw->plugin ? raw->plugin->getName() : String());
                                        o->setProperty("error", err);
                                        if (onEvent) onEvent("vst_loaded", var(o));
                                    });
                                });
                            } else if (!t->plugin && t->vstPath.isNotEmpty()) {
                                // previous load failed; do not retry on every sync
                            }
                        } else {
                            if (t->instType != "sf2") { t->editor.reset(); t->plugin.reset(); t->vstPath = ""; t->vstName = ""; }
                            t->instType = "sf2";
                            t->bank = (int) inst["bank"]; t->program = (int) inst["program"]; t->drums = (bool) inst["drums"];
                        }
                        rebuildSchedule(*t, tv["notes"], bpm, offset);
                        newMidi.push_back(std::move(t));
                    }
                }
            }
            // sf2 channels
            int ch = 0;
            for (auto& t : newMidi) {
                if (t->instType == "sf2") {
                    if (t->sfChannel < 0 || t->sfChannel >= 16) t->sfChannel = ch;
                    ch = jmax(ch, t->sfChannel + 1);
                    applySf2Program(*t);
                } else if (t->sfChannel >= 0 && synth) { tsf_channel_note_off_all(synth, t->sfChannel); t->sfChannel = -1; }
            }
            // tracks removed: editors closed by destruction
            audioTracks.swap(newAudio);
            midiTracks.swap(newMidi);
            anySolo = solo;
            updateEnd();
        }
        newMidi.clear();   // old tracks (plugins) destroyed outside the lock
        for (auto& l : pluginLoads) l();
        if (done) done({});
    };
    if (toLoad.empty()) { install({}); return; }
    std::thread([this, toLoad, install]() {
        std::map<String, std::shared_ptr<AudioBuffer<float>>> loaded;
        for (auto& p : toLoad) { auto b = loadAudioFile(p.second); if (b) loaded[p.first] = b; }
        MessageManager::callAsync([install, loaded]() { install(loaded); });
    }).detach();
}

var Engine::vstInfo() {
    const ScopedLock sl(lock);
    auto* o = new DynamicObject();
    auto* v = new DynamicObject();
    for (auto& t : midiTracks) if (t->plugin) v->setProperty(t->id, t->plugin->getName());
    o->setProperty("vst_tracks", var(v));
    return var(o);
}

// ---------------------------------------------------------------- transport
void Engine::play() { isPlaying = true; }
void Engine::stop() { isPlaying = false; allNotesOff(); }
void Engine::seek(double sec) {
    pos = (int64_t) jmax(0.0, sec * sr);
    const ScopedLock sl(lock);
    const int64_t p = pos.load();
    for (auto& t : midiTracks) {
        t->nextNote = (size_t) (std::lower_bound(t->notes.begin(), t->notes.end(), p, [](const NoteEvent& e, int64_t v) { return e.start < v; }) - t->notes.begin());
        t->active.clear();
    }
    allNotesOff();
}
void Engine::setLoop(bool on, double a, double b) {
    loopA = (int64_t) jmax(0.0, a * sr); loopB = (int64_t) jmax(0.0, b * sr);
    loopOn = on && loopB.load() > loopA.load();
}
bool Engine::audible(bool mute, bool solo) const { return !mute && (!anySolo.load() || solo); }

void Engine::allNotesOff() {
    const ScopedLock sl(lock);
    for (auto& t : midiTracks) {
        if (t->instType == "vst3") t->pending.addEvent(MidiMessage::allNotesOff(1), 0);
        else if (synth && t->sfChannel >= 0) tsf_channel_note_off_all(synth, t->sfChannel);
        t->active.clear();
    }
}

void Engine::preview(const String& trackId, int pitch, int vel, double dur) {
    const ScopedLock sl(lock);
    for (auto& t : midiTracks) if (t->id == trackId) {
        t->pending.addEvent(MidiMessage::noteOn(1, pitch, (uint8) jlimit(1, 127, vel)), 0);
        previewOffs.push_back({ clock.load() + (int64_t) (dur * sr), t.get(), pitch });
    }
}
void Engine::previewOff(const String& trackId, int pitch) {
    const ScopedLock sl(lock);
    for (auto& t : midiTracks) if (t->id == trackId) t->pending.addEvent(MidiMessage::noteOff(1, pitch), 0);
}

// ---------------------------------------------------------------- plugins
StringArray Engine::pluginNames(const String& path, String& error) {
    StringArray names;
    for (auto* fmt : formatManager.getFormats()) {
        OwnedArray<PluginDescription> types;
        fmt->findAllTypesForFile(types, path);
        for (auto* d : types) names.add(d->name);
    }
    if (names.isEmpty()) error = String(CharPointer_UTF8("плагин не распознан: ")) + path;
    return names;
}

void Engine::preparePlugin(MidiTrack& t) {
    if (!t.plugin) return;
    t.plugin->setRateAndBufferSizeDetails(sr, block);
    t.plugin->prepareToPlay(sr, block);
    t.scratch.setSize(jmax(2, t.plugin->getTotalNumOutputChannels()), jmax(block, 512) * 2);
}

void Engine::loadPlugin(const String& trackId, const String& path, const String& name, std::function<void(bool, String)> done) {
    PluginDescription chosen; bool found = false;
    for (auto* fmt : formatManager.getFormats()) {
        OwnedArray<PluginDescription> types;
        fmt->findAllTypesForFile(types, path);
        for (auto* d : types) {
            if (!found || (name.isNotEmpty() && d->name == name) || (name.isEmpty() && d->isInstrument && !chosen.isInstrument)) { chosen = *d; found = true; }
        }
    }
    if (!found) { done(false, String(CharPointer_UTF8("плагин не распознан (формат не поддерживается или сбой при сканировании): ")) + path); return; }
    logLine("loading plugin " + chosen.name + " from " + path);
    formatManager.createPluginInstanceAsync(chosen, sr, block, [this, trackId, done, path, name](std::unique_ptr<AudioPluginInstance> inst, const String& err) {
        if (!inst) { logLine("plugin load failed: " + err); done(false, err); return; }
        inst->enableAllBuses();
        {
            const ScopedLock sl(lock);
            for (auto& t : midiTracks) if (t->id == trackId) {
                t->instType = "vst3"; t->vstPath = path; t->vstName = name;
                if (t->stateB64.isNotEmpty()) {
                    MemoryOutputStream mo; Base64::convertFromBase64(mo, t->stateB64);
                    inst->setStateInformation(mo.getData(), (int) mo.getDataSize());
                }
                t->plugin = std::move(inst);
                preparePlugin(*t);
                logLine("plugin ready: " + t->plugin->getName() + " outs=" + String(t->plugin->getTotalNumOutputChannels()) + " latency=" + String(t->plugin->getLatencySamples()));
                done(true, {});
                return;
            }
        }
        done(false, String(CharPointer_UTF8("дорожка исчезла во время загрузки")));
    });
}

String Engine::pluginState(const String& trackId) {
    const ScopedLock sl(lock);
    for (auto& t : midiTracks) if (t->id == trackId && t->plugin) {
        MemoryBlock mb; t->plugin->getStateInformation(mb);
        return Base64::toBase64(mb.getData(), mb.getSize());
    }
    return {};
}

class EditorWindow : public DocumentWindow {
public:
    EditorWindow(const String& title, std::function<void()> onClose)
        : DocumentWindow(title, Colours::darkgrey, DocumentWindow::allButtons), closeFn(std::move(onClose)) {
        setUsingNativeTitleBar(true);
    }
    void closeButtonPressed() override { if (closeFn) closeFn(); }
    std::function<void()> closeFn;
};

bool Engine::showEditor(const String& trackId, String& error, std::function<void()> onClosed) {
    MidiTrack* track = nullptr;
    { const ScopedLock sl(lock); for (auto& t : midiTracks) if (t->id == trackId) track = t.get(); }
    if (!track || !track->plugin) { error = String(CharPointer_UTF8("на дорожке нет загруженного VST3")); return false; }
    if (track->editor) { track->editor->toFront(true); return true; }
    auto* ed = track->plugin->hasEditor() ? track->plugin->createEditorIfNeeded() : nullptr;
    if (!ed) { ed = new GenericAudioProcessorEditor(*track->plugin); }
    auto* win = new EditorWindow(track->plugin->getName() + String(CharPointer_UTF8(" — ")) + track->name, [this, trackId, onClosed]() {
        MessageManager::callAsync([this, trackId, onClosed]() {
            { const ScopedLock sl(lock); for (auto& t : midiTracks) if (t->id == trackId) t->editor.reset(); }
            if (onClosed) onClosed();
        });
    });
    win->setContentOwned(ed, true);
    win->setResizable(true, false);
    win->centreWithSize(jmax(300, ed->getWidth()), jmax(200, ed->getHeight()));
    win->setVisible(true);
    win->toFront(true);
    track->editor.reset(win);
    return true;
}

int Engine::closeEditors() {
    std::vector<std::function<void()>> closers;
    { const ScopedLock sl(lock); for (auto& t : midiTracks) if (t->editor) if (auto* w = dynamic_cast<EditorWindow*>(t->editor.get())) closers.push_back(w->closeFn); }
    for (auto& c : closers) c();
    return (int) closers.size();
}

bool Engine::editorOpen() const {
    const ScopedLock sl(lock);
    for (auto& t : midiTracks) if (t->editor) return true;
    return false;
}

void Engine::startTest(const String& trackId) {
    const ScopedLock sl(lock);
    for (auto& t : midiTracks) if (t->id == trackId) {
        t->testPeak = 0.0f; t->testing = true;
        t->pending.addEvent(MidiMessage::noteOn(1, t->drums ? 36 : 60, (uint8) 100), 0);
        previewOffs.push_back({ clock.load() + (int64_t) (0.5 * sr), t.get(), t->drums ? 36 : 60 });
    }
}
float Engine::testResult(const String& trackId) {
    const ScopedLock sl(lock);
    for (auto& t : midiTracks) if (t->id == trackId) { t->testing = false; return t->testPeak.load(); }
    return -1.0f;
}

double Engine::offlineRender(double seconds) {
    if (!deviceOk) { sr = 44100.0; block = 1024; const ScopedLock sl(lock); if (synth) tsf_set_output(synth, TSF_STEREO_INTERLEAVED, (int) sr, 0.0f); for (auto& t : midiTracks) if (t->plugin) preparePlugin(*t); }
    std::vector<float> L((size_t) block), R((size_t) block);
    float* outs[2] = { L.data(), R.data() };
    float peak = 0.0f;
    const int blocks = (int) (seconds * sr / block);
    for (int i = 0; i < blocks; ++i) {
        audioDeviceIOCallbackWithContext(nullptr, 0, outs, 2, block, AudioIODeviceCallbackContext());
        for (int k = 0; k < block; ++k) peak = jmax(peak, std::abs(L[(size_t) k]), std::abs(R[(size_t) k]));
    }
    return peak;
}

// ---------------------------------------------------------------- audio callback
void Engine::audioDeviceIOCallbackWithContext(const float* const*, int, float* const* out, int nOut, int n, const AudioIODeviceCallbackContext&) {
    for (int c = 0; c < nOut; ++c) FloatVectorOperations::clear(out[c], n);
    if (nOut < 1) return;
    if (mix.getNumSamples() < n) mix.setSize(2, n, false, false, true);
    mix.clear();
    const ScopedTryLock stl(lock);
    if (!stl.isLocked()) { clock += n; return; }   // message thread is restructuring: one silent block
    // preview note-offs by absolute clock
    const int64_t c0 = clock.load();
    for (size_t i = 0; i < previewOffs.size();) {
        if (previewOffs[i].at <= c0 + n) {
            auto* t = previewOffs[i].track;
            bool alive = false; for (auto& mt : midiTracks) if (mt.get() == t) alive = true;
            if (alive) t->pending.addEvent(MidiMessage::noteOff(1, previewOffs[i].pitch), 0);
            previewOffs.erase(previewOffs.begin() + (long) i);
        } else ++i;
    }
    if (isPlaying.load()) {
        int done = 0; int64_t p = pos.load();
        while (done < n) {
            int seg = n - done;
            if (loopOn.load() && loopB.load() > p) seg = (int) jmin((int64_t) seg, loopB.load() - p);
            AudioBuffer<float> sub(mix.getArrayOfWritePointers(), 2, done, seg);
            renderSegment(p, seg, sub);
            p += seg; done += seg;
            if (loopOn.load() && p >= loopB.load()) {
                p = loopA.load();
                for (auto& t : midiTracks) {
                    t->nextNote = (size_t) (std::lower_bound(t->notes.begin(), t->notes.end(), p, [](const NoteEvent& e, int64_t v) { return e.start < v; }) - t->notes.begin());
                    t->active.clear();
                    if (t->instType == "vst3") t->pending.addEvent(MidiMessage::allNotesOff(1), 0);
                    else if (synth && t->sfChannel >= 0) tsf_channel_note_off_all(synth, t->sfChannel);
                }
            } else if (p >= endSample.load()) {
                isPlaying = false;
                for (auto& t : midiTracks) { t->active.clear(); if (t->instType == "vst3") t->pending.addEvent(MidiMessage::allNotesOff(1), 0); else if (synth && t->sfChannel >= 0) tsf_channel_note_off_all(synth, t->sfChannel); }
                if (onEvent) MessageManager::callAsync([this]() { if (onEvent) onEvent("stopped", var()); });
                break;
            }
        }
        pos = p;
    } else {
        AudioBuffer<float> sub(mix.getArrayOfWritePointers(), 2, 0, n);
        renderSegment(-1, n, sub);   // idle: previews / tails only
    }
    clock += n;
    // master + gentle limiter
    const float m = master.load();
    float peak = 0.0f;
    for (int c = 0; c < 2; ++c) peak = jmax(peak, mix.getMagnitude(c, 0, n));
    peak *= m;
    if (peak > 0.98f) limiter = jmin(limiter, 0.98f / peak); else limiter = jmin(1.0f, limiter * 1.02f);
    for (int c = 0; c < nOut; ++c) {
        FloatVectorOperations::addWithMultiply(out[c], mix.getReadPointer(jmin(c, 1)), m * limiter, n);
        FloatVectorOperations::clip(out[c], out[c], -1.0f, 1.0f, n);
    }
}

// p < 0 means "transport idle": only pending (preview) events are processed
void Engine::renderSegment(int64_t p, int n, AudioBuffer<float>& out) {
    const bool running = p >= 0;
    // audio tracks
    if (running) for (auto& a : audioTracks) {
        if (!a->buffer || !audible(a->mute, a->solo)) continue;
        const int64_t len = a->buffer->getNumSamples();
        if (p >= len) continue;
        const int m = (int) jmin((int64_t) n, len - p);
        for (int c = 0; c < 2; ++c) out.addFrom(c, 0, *a->buffer, c, (int) p, m, a->volume);
    }
    // SoundFont tracks: sample-accurate sub-blocks
    if (synth) {
        struct Ev { int off; int ch; int pitch; int vel; bool on; };
        std::vector<Ev> evs;
        for (auto& t : midiTracks) {
            if (t->instType != "sf2" || t->sfChannel < 0) continue;
            const bool aud = audible(t->mute, t->solo);
            tsf_channel_set_volume(synth, t->sfChannel, aud ? t->volume : 0.0f);
            for (auto m : t->pending) {
                auto msg = m.getMessage();
                if (msg.isNoteOn()) evs.push_back({ 0, t->sfChannel, msg.getNoteNumber(), msg.getVelocity(), true });
                else if (msg.isNoteOff()) evs.push_back({ 0, t->sfChannel, msg.getNoteNumber(), 0, false });
                else if (msg.isAllNotesOff()) tsf_channel_note_off_all(synth, t->sfChannel);
            }
            t->pending.clear();
            if (!running) continue;
            for (size_t i = 0; i < t->active.size();) {
                if (t->active[i].end < p + n) { evs.push_back({ (int) jmax<int64_t>(0, t->active[i].end - p), t->sfChannel, t->active[i].pitch, 0, false }); t->active.erase(t->active.begin() + (long) i); }
                else ++i;
            }
            while (t->nextNote < t->notes.size() && t->notes[t->nextNote].start < p + n) {
                auto& e = t->notes[t->nextNote++];
                if (aud) { evs.push_back({ (int) jmax<int64_t>(0, e.start - p), t->sfChannel, e.pitch, e.vel, true }); t->active.push_back({ e.end, e.pitch }); }
            }
        }
        std::sort(evs.begin(), evs.end(), [](const Ev& a, const Ev& b) { return a.off < b.off || (a.off == b.off && !a.on && b.on); });
        if (sfBuf.size() < (size_t) n * 2) sfBuf.resize((size_t) n * 2);
        int cur = 0;
        auto renderTo = [&](int upto) {
            const int len = upto - cur;
            if (len <= 0) return;
            tsf_render_float(synth, sfBuf.data(), len, 0);
            float* L = out.getWritePointer(0) + cur; float* R = out.getWritePointer(1) + cur;
            for (int i = 0; i < len; ++i) { L[i] += sfBuf[(size_t) i * 2]; R[i] += sfBuf[(size_t) i * 2 + 1]; }
            cur = upto;
        };
        for (auto& e : evs) {
            renderTo(e.off);
            if (e.on) tsf_channel_note_on(synth, e.ch, e.pitch, (float) e.vel / 127.0f); else tsf_channel_note_off(synth, e.ch, e.pitch);
        }
        renderTo(n);
    }
    // VST3 tracks
    for (auto& t : midiTracks) {
        if (t->instType != "vst3" || !t->plugin) continue;
        const bool aud = audible(t->mute, t->solo);
        MidiBuffer midi;
        midi.addEvents(t->pending, 0, n, 0);
        t->pending.clear();
        if (running) {
            for (size_t i = 0; i < t->active.size();) {
                if (t->active[i].end < p + n) { midi.addEvent(MidiMessage::noteOff(1, t->active[i].pitch), (int) jmax<int64_t>(0, t->active[i].end - p)); t->active.erase(t->active.begin() + (long) i); }
                else ++i;
            }
            while (t->nextNote < t->notes.size() && t->notes[t->nextNote].start < p + n) {
                auto& e = t->notes[t->nextNote++];
                if (aud) { midi.addEvent(MidiMessage::noteOn(1, e.pitch, (uint8) jlimit(1, 127, e.vel)), (int) jmax<int64_t>(0, e.start - p)); t->active.push_back({ e.end, e.pitch }); }
            }
        }
        if (t->scratch.getNumSamples() < n) t->scratch.setSize(t->scratch.getNumChannels(), n, false, false, true);
        AudioBuffer<float> buf(t->scratch.getArrayOfWritePointers(), t->scratch.getNumChannels(), 0, n);
        buf.clear();
        if (!t->plugin->isSuspended()) t->plugin->processBlock(buf, midi);
        if (t->testing) { float pk = 0; for (int c = 0; c < jmin(2, buf.getNumChannels()); ++c) pk = jmax(pk, buf.getMagnitude(c, 0, n)); t->testPeak = jmax(t->testPeak.load(), pk); }
        if (aud) for (int c = 0; c < 2; ++c) out.addFrom(c, 0, buf, jmin(c, buf.getNumChannels() - 1), 0, n, t->volume);
    }
}

} // namespace sn
