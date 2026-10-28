"""Export project tracks to a Standard MIDI File (type 1) that FL Studio imports with one channel per track."""
from __future__ import annotations

from pathlib import Path

import mido

from .project import Project

PPQ = 960

_TRANSLIT = dict(zip("абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
                     ["a", "b", "v", "g", "d", "e", "yo", "zh", "z", "i", "y", "k", "l", "m", "n", "o", "p", "r", "s", "t",
                      "u", "f", "h", "ts", "ch", "sh", "sch", "", "y", "", "e", "yu", "ya"]))


def ascii_name(name: str) -> str:
    """MIDI meta strings are latin-1; transliterate Cyrillic so FL Studio shows readable track names."""
    out = []
    for ch in name:
        low = ch.lower()
        if low in _TRANSLIT:
            t = _TRANSLIT[low]
            out.append(t.capitalize() if ch.isupper() else t)
        elif ord(ch) < 256:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out) or "Track"


def export_midi(project: Project, path: Path, align_to_song_start: bool = True, selected_ids: set[str] | None = None) -> int:
    """Returns the number of exported notes."""
    mid = mido.MidiFile(type=1, ticks_per_beat=PPQ)
    tempo = mido.bpm2tempo(project.bpm)

    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("track_name", name=ascii_name(project.name), time=0))
    meta.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))
    meta.append(mido.MetaMessage("time_signature", numerator=project.beats_per_bar, denominator=4, time=0))
    meta.append(mido.MetaMessage("end_of_track", time=0))
    mid.tracks.append(meta)

    # If aligned to song start, beat 0 of the grid is at project.offset seconds -> shift in beats
    shift_beats = project.sec_to_beat(0.0) * -1.0 if align_to_song_start else 0.0  # = offset*bpm/60

    channel_iter = iter([c for c in range(16) if c != 9])
    count = 0
    for track in project.midi_tracks():
        if selected_ids is not None and track.id not in selected_ids:
            continue
        if not track.notes:
            continue
        if track.is_drums:
            channel = 9
        else:
            try:
                channel = next(channel_iter)
            except StopIteration:
                channel = 15
        mt = mido.MidiTrack()
        mt.append(mido.MetaMessage("track_name", name=ascii_name(track.name), time=0))
        program = int(track.instrument.get("program", 0)) if track.instrument.get("type") == "sf2" else 0
        if not track.is_drums:
            mt.append(mido.Message("program_change", channel=channel, program=max(0, min(127, program)), time=0))

        events: list[tuple[int, int, mido.Message]] = []  # (tick, order, msg) order: note_off before note_on at same tick
        for n in track.notes:
            start_beat = n.start + shift_beats
            end_beat = n.end + shift_beats
            if end_beat <= 0:
                continue
            start_beat = max(0.0, start_beat)
            on_tick = int(round(start_beat * PPQ))
            off_tick = max(on_tick + 1, int(round(end_beat * PPQ)))
            vel = max(1, min(127, int(n.velocity)))
            events.append((on_tick, 1, mido.Message("note_on", channel=channel, note=int(n.pitch), velocity=vel, time=0)))
            events.append((off_tick, 0, mido.Message("note_off", channel=channel, note=int(n.pitch), velocity=0, time=0)))
            count += 1
        events.sort(key=lambda e: (e[0], e[1]))
        last = 0
        for tick, _, msg in events:
            msg.time = tick - last
            last = tick
            mt.append(msg)
        mt.append(mido.MetaMessage("end_of_track", time=0))
        mid.tracks.append(mt)

    mid.save(str(path))
    return count
