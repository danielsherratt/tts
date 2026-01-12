export async function onRequestPost({ request, env }) {
  try {
    const { text, voice = "alloy", speed = 1 } = await request.json();

    if (!text || !text.trim()) {
      return new Response("Missing text", { status: 400 });
    }

    /* 1️⃣ Request WAV from OpenAI */
    const openaiRes = await fetch("https://api.openai.com/v1/audio/speech", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${env.OPENAI_API_KEY}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model: "gpt-4o-mini-tts",
        input: text,
        voice,
        format: "wav",
        speed
      })
    });

    if (!openaiRes.ok) {
      return new Response(await openaiRes.text(), {
        status: openaiRes.status
      });
    }

    const inputBuffer = await openaiRes.arrayBuffer();

    /* 2️⃣ Decode WAV → AudioBuffer */
    const audioCtx = new AudioContext();
    const decoded = await audioCtx.decodeAudioData(inputBuffer);

    /* 3️⃣ Convert to mono */
    const mono = audioCtx.createBuffer(
      1,
      decoded.length,
      decoded.sampleRate
    );
    const monoData = mono.getChannelData(0);

    for (let i = 0; i < decoded.length; i++) {
      let sum = 0;
      for (let c = 0; c < decoded.numberOfChannels; c++) {
        sum += decoded.getChannelData(c)[i];
      }
      monoData[i] = sum / decoded.numberOfChannels;
    }

    /* 4️⃣ Resample → 8 kHz */
    const targetRate = 8000;
    const ratio = decoded.sampleRate / targetRate;
    const newLength = Math.floor(mono.length / ratio);
    const resampled = new Float32Array(newLength);

    for (let i = 0; i < newLength; i++) {
      resampled[i] = monoData[Math.floor(i * ratio)] || 0;
    }

    /* 5️⃣ Encode PCM16 WAV */
    const wavBuffer = encodeWavPCM16(resampled, targetRate);

    return new Response(wavBuffer, {
      headers: {
        "Content-Type": "audio/wav",
        "Content-Disposition": 'attachment; filename="tts-3cx.wav"',
        "Cache-Control": "no-store"
      }
    });

  } catch (err) {
    return new Response(err?.message || "Server error", { status: 500 });
  }
}

/* ============================
   WAV PCM16 encoder (mono)
   ============================ */
function encodeWavPCM16(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  let offset = 0;

  const writeString = s => {
    for (let i = 0; i < s.length; i++) {
      view.setUint8(offset++, s.charCodeAt(i));
    }
  };

  writeString("RIFF");
  view.setUint32(offset, 36 + samples.length * 2, true); offset += 4;
  writeString("WAVE");
  writeString("fmt ");
  view.setUint32(offset, 16, true); offset += 4;
  view.setUint16(offset, 1, true); offset += 2;        // PCM
  view.setUint16(offset, 1, true); offset += 2;        // mono
  view.setUint32(offset, sampleRate, true); offset += 4;
  view.setUint32(offset, sampleRate * 2, true); offset += 4;
  view.setUint16(offset, 2, true); offset += 2;
  view.setUint16(offset, 16, true); offset += 2;
  writeString("data");
  view.setUint32(offset, samples.length * 2, true); offset += 4;

  for (let i = 0; i < samples.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }

  return buffer;
}
