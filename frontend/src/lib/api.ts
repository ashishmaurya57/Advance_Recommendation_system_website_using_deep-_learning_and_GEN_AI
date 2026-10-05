export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

type Body = FormData | Record<string, unknown> | undefined;

async function request<T>(method: string, path: string, body?: Body): Promise<T> {
  const headers: Record<string, string> = {};
  const init: RequestInit = { method, credentials: "include", headers };
  if (body instanceof FormData) {
    init.body = body;
  } else if (body !== undefined) {
    init.body = JSON.stringify(body);
    headers["Content-Type"] = "application/json";
  }

  let res: Response;
  try {
    res = await fetch(`/api${path}`, init);
  } catch {
    throw new ApiError(0, "Can't reach the server. Is the backend running?");
  }
  if (!res.ok) {
    let message =
      res.status === 404 && path !== "/auth/me"
        ? `The server couldn't find ${path} (404). Is the backend deployed and the /api rewrite set up?`
        : res.statusText || `Request failed (${res.status}).`;
    try {
      const data = await res.json();
      if (typeof data.detail === "string") message = data.detail;
      else if (Array.isArray(data.detail) && data.detail[0]?.msg) {
        const d = data.detail[0];
        const field = Array.isArray(d.loc) ? d.loc[d.loc.length - 1] : "";
        message = field ? `${field}: ${d.msg}` : d.msg;
      }
    } catch {
      // non-JSON error body
    }
    throw new ApiError(res.status, message);
  }
  return res.status === 204 ? (undefined as T) : res.json();
}

export const api = {
  get: <T>(path: string) => request<T>("GET", path),
  post: <T>(path: string, body?: Body) => request<T>("POST", path, body),
  put: <T>(path: string, body?: Body) => request<T>("PUT", path, body),
  delete: <T>(path: string) => request<T>("DELETE", path),
};
