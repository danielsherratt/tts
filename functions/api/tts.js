// /functions/api/tts.js
export async function onRequestPost({ request, env }) {
  try {
    const { text, voice = "alloy", speed = 1 } = await request.json();

    if (!text || !text.trim()) {
      return new Response("Missing text", { status: 400 });
    }

    // 1) Get WAV from OpenAI
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
        format: "wav",
        speed,
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

    // 2) Parse WAV to float mono samples (Worker-safe, no AudioContext)
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

/* ============================
   WAV decode: PCM16/PCM8/Float32
   Returns mono Float32Array
   ============================ */
function decodeWavToFloatMono(arrayBuffer) {
  const view = new DataView(arrayBuffer);

  const riff = readFourCC(view, 0);
  const wave = readFourCC(view, 8);
  if (riff !== "RIFF" || wave !== "WAVE") {
    throw new Error("Invalid WAV (missing RIFF/WAVE)");
  }

  // Walk chunks to find "fmt " and "data"
  let offset = 12;
  let fmt = null;
  let dataOffset = null;
  let dataSize = null;

  while (offset + 8 <= view.byteLength) {
    const id = readFourCC(view, offset);
    const size = view.getUint32(offset + 4, true);
    const chunkStart = offset + 8;

    if (id === "fmt ") {
      fmt = parseFmtChunk(view, chunkStart, size);
    } else if (id === "data") {
      dataOffset = chunkStart;
      dataSize = size;
    }

    // chunks are padded to even sizes
    offset = chunkStart + size + (size % 2);
    if (fmt && dataOffset != null) break;
  }

  if (!fmt) throw new Error("Invalid WAV (missing fmt chunk)");
  if (dataOffset == null || dataSize == null) throw new Error("Invalid WAV (missing data chunk)");

  const { audioFormat, numChannels, sampleRate, bitsPerSample } = fmt;

  // Supported: PCM (1) and IEEE float (3)
  if (audioFormat !== 1 && audioFormat !== 3) {
    throw new Error(`Unsupported WAV format: ${audioFormat}`);
  }

  // Read samples into mono float [-1..1]
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

  const audioFormat = view.getUint16(start + 0, true);
  const numChannels = view.getUint16(start + 2, true);
  const sampleRate = view.getUint32(start + 4, true);
  // const byteRate = view.getUint32(start + 8, true);
  // const blockAlign = view.getUint16(start + 12, true);
  const bitsPerSample = view.getUint16(start + 14, true);

  return { audioFormat, numChannels, sampleRate, bitsPerSample };
}

function readSampleAsFloat(view, offset, audioFormat, bitsPerSample) {
  // PCM
  if (audioFormat === 1) {
    if (bitsPerSample === 16) {
      const s = view.getInt16(offset, true);
      return s < 0 ? s / 32768 : s / 32767;
    }
    if (bitsPerSample === 8) {
      const s = view.getUint8(offset); // 0..255
      return (s - 128) / 128;
    }
    if (bitsPerSample === 24) {
      // 24-bit little endian signed
      const b0 = view.getUint8(offset);
      const b1 = view.getUint8(offset + 1);
      const b2 = view.getUint8(offset + 2);
      let v = (b2 << 16) | (b1 << 8) | b0;
      if (v & 0x800000) v |= 0xff000000; // sign extend
      // now v is signed 32-bit with 24-bit value
      return Math.max(-1, Math.min(1, v / 8388608));
    }
    throw new Error(`Unsupported PCM bit depth: ${bitsPerSample}`);
  }

  // IEEE float
  if (audioFormat === 3) {
    if (bitsPerSample === 32) {
      const f = view.getFloat32(offset, true);
      return Math.max(-1, Math.min(1, f));
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

/* ============================
   Resample (linear interpolation)
   ============================ */
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

/* ============================
   WAV encode PCM16 (mono)
   ============================ */
function encodeWavPCM16(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  let offset = 0;

  const writeString = (s) => {
    for (let i = 0; i < s.length; i++) view.setUint8(offset++, s.charCodeAt(i));
  };

  writeString("RIFF");
  view.setUint32(offset, 36 + samples.length * 2, true); offset += 4;
  writeString("WAVE");

  writeString("fmt ");
  view.setUint32(offset, 16, true); offset += 4;     // PCM fmt size
  view.setUint16(offset, 1, true); offset += 2;      // PCM
  view.setUint16(offset, 1, true); offset += 2;      // mono
  view.setUint32(offset, sampleRate, true); offset += 4;
  view.setUint32(offset, sampleRate * 2, true); offset += 4; // byteRate = sr * blockAlign
  view.setUint16(offset, 2, true); offset += 2;      // blockAlign = channels * bytesPerSample
  view.setUint16(offset, 16, true); offset += 2;     // bitsPerSample

  writeString("data");
  view.setUint32(offset, samples.length * 2, true); offset += 4;

  for (let i = 0; i < samples.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }

  return buffer;
}
