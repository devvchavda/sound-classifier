// Runs on the browser's audio thread: hands every block of microphone samples
// (128 at a time, at the microphone's own rate) to the live page, unchanged.
class MicTap extends AudioWorkletProcessor {
  process(inputs) {
    const samples = inputs[0][0]; // first input, first (mono) channel
    if (samples) this.port.postMessage(samples.slice(0)); // a copy: the browser reuses the buffer
    return true; // keep running
  }
}
registerProcessor("mic-tap", MicTap);
