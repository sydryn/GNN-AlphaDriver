export async function POST(request: Request) {
  const baseUrl = process.env.ALPHADRIVER_API_URL ?? "http://127.0.0.1:8000";
  try {
    const upstream = await fetch(new URL("/api/v1/predict", baseUrl), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: await request.text(),
      cache: "no-store",
      signal: AbortSignal.any([request.signal, AbortSignal.timeout(60_000)]),
    });
    const headers = new Headers({ "Content-Type": upstream.headers.get("content-type") ?? "application/json" });
    const retryAfter = upstream.headers.get("retry-after");
    if (retryAfter) headers.set("Retry-After", retryAfter);
    return new Response(upstream.body, { status: upstream.status, headers });
  } catch {
    return Response.json(
      { code: "api_unavailable", message: "The prediction API is unavailable. Check that it is running and try again." },
      { status: 502 },
    );
  }
}
