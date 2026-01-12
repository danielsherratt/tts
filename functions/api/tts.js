export async function onRequestPost({ request, env }) {
  try {
    const { text, voice = "alloy", format = "wav", speed = 1 } = await request.json();

    if (!text || !text.trim()) {
      return new Response("Missing text", { status: 400 });
    }

    const openaiResponse = await fetch(
      "https://api.openai.com/v1/audio/speech",
      {
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
      }
    );

    if (!openaiResponse.ok) {
      const err = await openaiResponse.text();
      return new Response(err, {
        status: openaiResponse.status,
        headers: { "Content-Type": "text/plain" }
      });
    }

    const audioBuffer = await openaiResponse.arrayBuffer();

    return new Response(audioBuffer, {
      headers: {
        "Content-Type": "audio/wav",
        "Content-Disposition": 'attachment; filename="tts-3cx.wav"',
        "Cache-Control": "no-store"
      }
    });

  } catch (err) {
    return new Response(
      err?.message || "Internal Server Error",
      { status: 500 }
    );
  }
}
