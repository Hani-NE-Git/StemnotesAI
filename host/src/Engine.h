// StemNotes native audio engine: audio tracks, SoundFont (TinySoundFont) and VST3 instruments in real time.
#pragma once
#include <juce_audio_utils/juce_audio_utils.h>
#include <juce_audio_processors/juce_audio_processors.h>
#include <atomic>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <memory>
#include <vector>

struct tsf;

namespace sn {

struct NoteEvent { int64_t start = 0, end = 0; int pitch = 60, vel = 100; };
struct ActiveNote { int64_t end; int pitch; };

struct AudioTrack {
    juce::String id, path;
    bool mute = false, solo = false;
    float volume = 0.8f;
    std::shared_ptr<juce::AudioBuffer<float>> buffer;   // stereo, engine sample rate
};

struct MidiTrack {
    juce::String id, name;
    bool mute = false, solo = false;
    float volume = 0.8f;
    // instrument
    juce::String instType = "sf2";          // "sf2" | "vst3"
    int bank = 0, program = 0; bool drums = false;
    juce::String vstPath, vstName, stateB64;
    std::unique_ptr<juce::AudioPluginInstance> plugin;
    std::unique_ptr<juce::DocumentWindow> editor;
    int sfChannel = -1;
    // schedule
    std::vector<NoteEvent> notes;           // sorted by start
    size_t nextNote = 0;
    std::vector<ActiveNote> active;
    juce::MidiBuffer pending;               // preview / immediate events for the next block (message thread -> callback)
    juce::AudioBuffer<float> scratch;
    std::vector<std::pair<int, int>> outBuses;   // (first channel, channel count) of every enabled output bus
    std::atomic<float> testPeak { 0.0f };
    bool testing = false;
};

class Engine : public juce::AudioIODeviceCallback {
public:
    Engine();
    ~Engine() override;

    // device
    juce::String start(const juce::String& deviceName, int bufferSize, juce::String& error);
    void stopDevice();
    void shutdown();      // stop device, close editors, unload plugins (before process exit)
    juce::var listDevices();
    double sampleRate() const { return sr; }
    int blockSize() const { return block; }
    bool available() const { return deviceOk; }

    // content (message thread)
    bool loadSoundFont(const juce::File& f, juce::String& error);
    void applyProject(const juce::var& project, const juce::File& dir, std::function<void(juce::String)> done);
    juce::var vstInfo();

    // transport (any thread)
    void play();
    void stop();
    void seek(double sec);
    void setLoop(bool on, double a, double b);
    void setMaster(float v) { master = v; }
    double position() const { return (double) pos.load() / sr; }
    bool playing() const { return isPlaying.load(); }
    double endSec() const { return (double) endSample.load() / sr; }

    // notes
    void preview(const juce::String& trackId, int pitch, int vel, double dur);
    void previewOff(const juce::String& trackId, int pitch);

    // plugins (message thread)
    juce::StringArray pluginNames(const juce::String& path, juce::String& error);
    void loadPlugin(const juce::String& trackId, const juce::String& path, const juce::String& name,
                    std::function<void(bool, juce::String)> done);
    juce::String pluginState(const juce::String& trackId);
    bool showEditor(const juce::String& trackId, juce::String& error, std::function<void()> onClosed);
    int closeEditors();
    bool editorOpen() const;
    void startTest(const juce::String& trackId);
    float testResult(const juce::String& trackId);

    // test helper: run the audio callback without a device (sets 44100/1024 if no device is open)
    double offlineRender(double seconds);

    // AudioIODeviceCallback
    void audioDeviceIOCallbackWithContext(const float* const* in, int nIn, float* const* out, int nOut, int n,
                                          const juce::AudioIODeviceCallbackContext&) override;
    void audioDeviceAboutToStart(juce::AudioIODevice* device) override;
    void audioDeviceStopped() override;

    std::function<void(const juce::String&, const juce::var&)> onEvent;   // name, payload (message thread)

private:
    void renderSegment(int64_t p, int n, juce::AudioBuffer<float>& out);
    void allNotesOff();
    void rebuildSchedule(MidiTrack& t, const juce::var& notes, double bpm, double offset);
    void updateEnd();
    bool audible(bool mute, bool solo) const;
    void applySf2Program(MidiTrack& t);
    void preparePlugin(MidiTrack& t);
    std::shared_ptr<juce::AudioBuffer<float>> loadAudioFile(const juce::File& f);

    juce::AudioDeviceManager deviceManager;
    juce::AudioPluginFormatManager formatManager;
    juce::AudioFormatManager audioFormats;
    juce::CriticalSection lock;                 // protects track vectors & plugin pointers
    std::vector<std::unique_ptr<AudioTrack>> audioTracks;
    std::vector<std::unique_ptr<MidiTrack>> midiTracks;
    std::map<juce::String, std::shared_ptr<juce::AudioBuffer<float>>> audioCache;   // path -> buffer
    tsf* synth = nullptr;
    juce::File soundfont;
    std::vector<float> sfBuf;
    double sr = 44100.0; int block = 1024; bool deviceOk = false;
    std::atomic<int64_t> pos { 0 }, endSample { 441000 }, loopA { 0 }, loopB { 0 }, clock { 0 };
    std::atomic<bool> isPlaying { false }, loopOn { false }, anySolo { false };
    std::atomic<float> master { 0.9f };
    float limiter = 1.0f;
    double lastBpm = 120.0, lastOffset = 0.0;
    struct PreviewOff { int64_t at; MidiTrack* track; int pitch; };
    std::vector<PreviewOff> previewOffs;
    juce::AudioBuffer<float> mix;
};

} // namespace sn
