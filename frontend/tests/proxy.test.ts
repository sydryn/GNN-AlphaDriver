import { expect, it, vi } from "vitest";
import { POST } from "../app/api/predict/route";

it("forwards the upstream status and request body", async () => {
  const fetcher = vi.spyOn(globalThis, "fetch").mockResolvedValue(Response.json({ code: "provider_busy" }, { status: 503 }));
  const response = await POST(new Request("http://localhost/api/predict", { method: "POST", body: "{}" }));
  expect(fetcher).toHaveBeenCalledWith(expect.any(URL), expect.objectContaining({ body: "{}", method: "POST" }));
  expect(response.status).toBe(503);
  expect(await response.json()).toEqual({ code: "provider_busy" });
});

it("returns a structured error when the upstream server is unreachable", async () => {
  vi.spyOn(globalThis, "fetch").mockRejectedValue(new TypeError("Connection refused"));
  const response = await POST(new Request("http://localhost/api/predict", { method: "POST", body: "{}" }));
  expect(response.status).toBe(502);
  expect(await response.json()).toMatchObject({ code: "api_unavailable" });
});
