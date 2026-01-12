export async function onRequestPost({ request, env }) {
  try {
    const { text, voice = "alloy", speed = 1 } = await request.json();

    if (!text || !text.trim()) {
      return new Response("Missing text", { status: 400 });
    }

    // 1) Get WAV from OpenAI (IMPORTANT: response_format, not format)
    const openaiRes = await fetch("https://api.openai.com/v1/audio/speech", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${env.OPENAI_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "gpt-4o-mini-tts",
        input: text,
        voice,
        speed,
        response_format: "wav", // ✅ correct param name
        // stream_format: "audio", // optional
      }),
    });

    if (!openaiRes.ok) {
      const err = await openaiRes.text().catch(() => "");
      return new Response(err || `OpenAI error (${openaiRes.status})`, {
        status: openaiRes.status,
        headers: { "Content-Type": "text/plain" },
      });
    }

    const wavIn = await openaiRes.arrayBuffer();

    // Quick sanity check (helps with debugging)
    const head = new Uint8Array(wavIn.slice(0, 12));
    const sig = String.fromCharCode(head[0], head[1], head[2], head[3]) + "|" +
                String.fromCharCode(head[8], head[9], head[10], head[11]);
    if (sig !== "RIFF|WAVE") {
      // If this triggers, OpenAI didn’t return WAV (usually means response_format wasn’t applied)
      return new Response(`Expected WAV (RIFF/WAVE) but got: ${sig}`, {
        status: 500,
        headers: { "Content-Type": "text/plain" },
      });
    }

    // 2) Decode WAV → mono float samples
    const decoded = decodeWavToFloatMono(wavIn);

    // 3) Resample to 8 kHz
    const targetRate = 8000;
    const mono8k = resampleLinear(decoded.samples, decoded.sampleRate, targetRate);

    // 4) Encode as PCM16 WAV (mono, 8kHz, 16-bit)
    const wavOut = encodeWavPCM16(mono8k, targetRate);

    return new Response(wavOut, {
      headers: {
        "Content-Type": "audio/wav",
        "Content-Disposition": 'attachment; filename="tts-3cx.wav"',
        "Cache-Control": "no-store",
      },
    });
  } catch (err) {
    return new Response(err?.message || "Server error", { status: 500 });
  }
}

/* ===== WAV decode: PCM16/PCM8/PCM24/Float32 -> mono Float32Array ===== */
function decodeWavToFloatMono(arrayBuffer) {
  const view = new DataView(arrayBuffer);

  const riff = readFourCC(view, 0);
  const wave = readFourCC(view, 8);
  if (riff !== "RIFF" || wave !== "WAVE") {
    throw new Error("Invalid WAV (missing RIFF/WAVE)");
  }

  let offset = 12;
  let fmt = null;
  let dataOffset = null;
  let dataSize = null;

  while (offset + 8 <= view.byteLength) {
    const id = readFourCC(view, offset);
    const size = view.getUint32(offset + 4, true);
    const chunkStart = offset + 8;

    if (id === "fmt ") fmt = parseFmtChunk(view, chunkStart, size);
    if (id === "data") { dataOffset = chunkStart; dataSize = size; }

    offset = chunkStart + size + (size % 2);
    if (fmt && dataOffset != null) break;
  }

  if (!fmt) throw new Error("Invalid WAV (missing fmt chunk)");
  if (dataOffset == null || dataSize == null) throw new Error("Invalid WAV (missing data chunk)");

  const { audioFormat, numChannels, sampleRate, bitsPerSample } = fmt;

  if (audioFormat !== 1 && audioFormat !== 3) {
    throw new Error(`Unsupported WAV format: ${audioFormat}`);
  }

  const bytesPerSample = bitsPerSample / 8;
  const frameSize = bytesPerSample * numChannels;
  const frameCount = Math.floor(dataSize / frameSize);

  const mono = new Float32Array(frameCount);
  let p = dataOffset;

  for (let i = 0; i < frameCount; i++) {
    let sum = 0;
    for (let ch = 0; ch < numChannels; ch++) {
      const sampleOffset = p + ch * bytesPerSample;
      sum += readSampleAsFloat(view, sampleOffset, audioFormat, bitsPerSample);
    }
    mono[i] = sum / numChannels;
    p += frameSize;
  }

  return { samples: mono, sampleRate };
}

function parseFmtChunk(view, start, size) {
  if (size < 16) throw new Error("Invalid fmt chunk");
  return {
    audioFormat: view.getUint16(start + 0, true),
    numChannels: view.getUint16(start + 2, true),
    sampleRate: view.getUint32(start + 4, true),
    bitsPerSample: view.getUint16(start + 14, true),
  };
}

function readSampleAsFloat(view, offset, audioFormat, bitsPerSample) {
  if (audioFormat === 1) { // PCM
    if (bitsPerSample === 16) {
      const s = view.getInt16(offset, true);
      return s < 0 ? s / 32768 : s / 32767;
    }
    if (bitsPerSample === 8) {
      return (view.getUint8(offset) - 128) / 128;
    }
    if (bitsPerSample === 24) {
      const b0 = view.getUint8(offset);
      const b1 = view.getUint8(offset + 1);
      const b2 = view.getUint8(offset + 2);
      let v = (b2 << 16) | (b1 << 8) | b0;
      if (v & 0x800000) v |= 0xff000000;
      return Math.max(-1, Math.min(1, v / 8388608));
    }
    throw new Error(`Unsupported PCM bit depth: ${bitsPerSample}`);
  }

  if (audioFormat === 3) { // IEEE float
    if (bitsPerSample === 32) {
      return Math.max(-1, Math.min(1, view.getFloat32(offset, true)));
    }
    throw new Error(`Unsupported float bit depth: ${bitsPerSample}`);
  }

  throw new Error("Unsupported WAV format");
}

function readFourCC(view, offset) {
  return String.fromCharCode(
    view.getUint8(offset),
    view.getUint8(offset + 1),
    view.getUint8(offset + 2),
    view.getUint8(offset + 3)
  );
}

/* ===== Resample (linear) ===== */
function resampleLinear(input, inRate, outRate) {
  if (inRate === outRate) return input;
  const ratio = inRate / outRate;
  const outLength = Math.max(1, Math.floor(input.length / ratio));
  const output = new Float32Array(outLength);

  for (let i = 0; i < outLength; i++) {
    const x = i * ratio;
    const x0 = Math.floor(x);
    const x1 = Math.min(x0 + 1, input.length - 1);
    const t = x - x0;
    output[i] = input[x0] * (1 - t) + input[x1] * t;
  }
  return output;
}

/* ===== Encode WAV PCM16 mono ===== */
function encodeWavPCM16(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  let offset = 0;
  const writeString = (s) => { for (let i = 0; i < s.length; i++) view.setUint8(offset++, s.charCodeAt(i)); };

  writeString("RIFF");
  view.setUint32(offset, 36 + samples.length * 2, true); offset += 4;
  writeString("WAVE");

  writeString("fmt ");
  view.setUint32(offset, 16, true); offset += 4;
  view.setUint16(offset, 1, true); offset += 2;  // PCM
  view.setUint16(offset, 1, true); offset += 2;  // Mono
  view.setUint32(offset, sampleRate, true); offset += 4;
  view.setUint32(offset, sampleRate * 2, true); offset += 4;
  view.setUint16(offset, 2, true); offset += 2;  // blockAlign
  view.setUint16(offset, 16, true); offset += 2; // 16-bit

  writeString("data");
  view.setUint32(offset, samples.length * 2, true); offset += 4;

  for (let i = 0; i < samples.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }

  return buffer;
}
