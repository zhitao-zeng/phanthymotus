# SenseVoice homophone replacement

SenseVoice uses greedy CTC decoding and cannot consume the transducer hotword
graph. These assets enable sherpa-onnx's phrase-level homophone replacer after
decoding.

`phrases.txt` intentionally contains only long, unambiguous robot commands and
the product wake phrase. `lexicon.txt` is the required subset of sherpa-onnx's
official `hr-files/lexicon.txt`; `replace.fst` is compiled from the phrase list.
This resource does not change acoustic scores or add candidates that greedy
decoding failed to emit.
