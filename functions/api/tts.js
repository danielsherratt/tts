export async function onRequestPost({ request, env }) {
  try {
    const { text, voice = "alloy", speed = 1 } = await request.json();

    if (!text || !text.trim()) {
      return new Response("Missing text", { status: 400 });
    }

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
        response_format: "mp3"
      }),
    });

    if (!openaiRes.ok) {
      const err = await openaiRes.text().catch(() => "");
      return new Response(err || `OpenAI error (${openaiRes.status})`, {
        status: openaiRes.status,
        headers: { "Content-Type": "text/plain" },
      });
    }

    const mp3 = await openaiRes.arrayBuffer();

    return new Response(mp3, {
      headers: {
        "Content-Type": "audio/mpeg",
        "Content-Disposition": 'attachment; filename="tts.mp3"',
        "Cache-Control": "no-store",
      },
    });
  } catch (err) {
    return new Response(err?.message || "Server error", { status: 500 });
  }
}
