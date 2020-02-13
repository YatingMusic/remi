import miditoolkit
import numpy as np

class MIDIChord(object):
    def __init__(self):
        # define pitch classes
        self.PITCH_CLASSES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        # define chord maps (required)
        self.CHORD_MAPS = {'maj': [0, 4],
                           'min': [0, 3],
                           'dim': [0, 3, 6],
                           'aug': [0, 4, 8],
                           'dom': [0, 4, 7, 10]}
        # define chord insiders (+1)
        self.CHORD_INSIDERS = {'maj': [7],
                               'min': [7],
                               'dim': [9],
                               'aug': [],
                               'dom': []}
        # define chord outsiders (-1)
        self.CHORD_OUTSIDERS_1 = {'maj': [2, 5, 9],
                                  'min': [2, 5, 8],
                                  'dim': [2, 5, 10],
                                  'aug': [2, 5, 9],
                                  'dom': [2, 5, 9]}
        # define chord outsiders (-2)
        self.CHORD_OUTSIDERS_2 = {'maj': [1, 3, 6, 8, 10],
                                  'min': [1, 4, 6, 9, 11],
                                  'dim': [1, 4, 7, 8, 11],
                                  'aug': [1, 3, 6, 7, 10],
                                  'dom': [1, 3, 6, 8, 11]}

    def note2pianoroll(self, notes, max_tick, ticks_per_beat):
        return miditoolkit.pianoroll.parser.notes2pianoroll(
                note_stream_ori=notes,
                max_tick=max_tick,
                ticks_per_beat=ticks_per_beat)

    def sequencing(self, chroma):
        candidates = {}
        for index in range(len(chroma)):
            if chroma[index]:
                root_note = index
                _chroma = np.roll(chroma, -root_note)
                sequence = np.where(_chroma == 1)[0]
                candidates[root_note] = list(sequence)
        return candidates

    def scoring(self, candidates):
        scores = {}
        qualities = {}
        for root_note, sequence in candidates.items():
            if 3 not in sequence and 4 not in sequence:
                scores[root_note] = -100
                qualities[root_note] = 'None'
            elif 3 in sequence and 4 in sequence:
                scores[root_note] = -100
                qualities[root_note] = 'None'
            else:
                # decide quality
                if 3 in sequence:
                    if 6 in sequence:
                        quality = 'dim'
                    else:
                        quality = 'min'
                elif 4 in sequence:
                    if 8 in sequence:
                        quality = 'aug'
                    else:
                        if 7 in sequence and 10 in sequence:
                            quality = 'dom'
                        else:
                            quality = 'maj'
                # decide score
                maps = self.CHORD_MAPS.get(quality)
                _notes = [n for n in sequence if n not in maps]
                score = 0
                for n in _notes:
                    if n in self.CHORD_OUTSIDERS_1.get(quality):
                        score -= 1
                    elif n in self.CHORD_OUTSIDERS_2.get(quality):
                        score -= 2
                    elif n in self.CHORD_INSIDERS.get(quality):
                        score += 1
                scores[root_note] = score
                qualities[root_note] = quality
        return scores, qualities

    def find_chord(self, pianoroll):
        chroma = miditoolkit.pianoroll.utils.tochroma(pianoroll=pianoroll)
        chroma = np.sum(chroma, axis=0)
        chroma = np.array([1 if c else 0 for c in chroma])
        if np.sum(chroma) == 0:
            return 'N', 'N', 'N', 0
        else:
            candidates = self.sequencing(chroma=chroma)
            scores, qualities = self.scoring(candidates=candidates)
            # bass note
            sorted_notes = []
            for i, v in enumerate(np.sum(pianoroll, axis=0)):
                if v > 0:
                    sorted_notes.append(int(i%12))
            bass_note = sorted_notes[0]
            # root note
            __root_note = []
            _max = max(scores.values())
            for _root_note, score in scores.items():
                if score == _max:
                    __root_note.append(_root_note)
            if len(__root_note) == 1:
                root_note = __root_note[0]
            else:
                #TODO: what should i do
                for n in sorted_notes:
                    if n in __root_note:
                        root_note = n
                        break
            # quality
            quality = qualities.get(root_note)
            sequence = candidates.get(root_note)
            # score
            score = scores.get(root_note)
            return self.PITCH_CLASSES[root_note], quality, self.PITCH_CLASSES[bass_note], score

    def greedy(self, candidates, max_tick, min_length):
        chords = []
        # start from 0
        start_tick = 0
        while start_tick < max_tick:
            _candidates = candidates.get(start_tick)
            _candidates = sorted(_candidates.items(), key=lambda x: (x[1][-1], x[0]))
            # choose
            end_tick, (root_note, quality, bass_note, _) = _candidates[-1]
            if root_note == bass_note:
                chord = '{}:{}'.format(root_note, quality)
            else:
                chord = '{}:{}/{}'.format(root_note, quality, bass_note)
            chords.append([start_tick, end_tick, chord])
            start_tick = end_tick
        # remove :None
        temp = chords
        while ':None' in temp[0][-1]:
            try:
                temp[1][0] = temp[0][0]
                del temp[0]
            except:
                print('NO CHORD')
                return []
        temp2 = []
        for chord in temp:
            if ':None' not in chord[-1]:
                temp2.append(chord)
            else:
                temp2[-1][1] = chord[1]
        return temp2

    def extract(self, notes):
        # read
        max_tick = max([n.end for n in notes])
        ticks_per_beat = 480
        pianoroll = self.note2pianoroll(
            notes=notes, 
            max_tick=max_tick, 
            ticks_per_beat=ticks_per_beat)
        # get lots of candidates
        candidates = {}
        # the shortest: 2 beat, longest: 4 beat
        for interval in [4, 2]:
            for start_tick in range(0, max_tick, ticks_per_beat):
                # set target pianoroll
                end_tick = int(ticks_per_beat * interval + start_tick)
                if end_tick > max_tick:
                    end_tick = max_tick
                _pianoroll = pianoroll[start_tick:end_tick, :]
                # find chord
                root_note, quality, bass_note, score = self.find_chord(pianoroll=_pianoroll)
                # save
                if start_tick not in candidates:
                    candidates[start_tick] = {}
                    candidates[start_tick][end_tick] = (root_note, quality, bass_note, score)
                else:
                    if end_tick not in candidates[start_tick]:
                        candidates[start_tick][end_tick] = (root_note, quality, bass_note, score)
        # greedy
        chords = self.greedy(candidates=candidates, 
                             max_tick=max_tick, 
                             min_length=ticks_per_beat)
        return chords
