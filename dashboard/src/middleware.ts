import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

const PUBLIC_PATHS = ["/login", "/signup", "/api/auth"];

const STATIC_PREFIXES = ["_next", "/favicon", "/fonts", "/images"];

function isPublicPath(pathname: string): boolean {
  if (STATIC_PREFIXES.some((prefix) => pathname.startsWith(prefix))) {
    return true;
  }
  if (pathname.startsWith("/api/auth")) {
    return true;
  }
  return PUBLIC_PATHS.some(
    (path) => pathname === path || pathname.startsWith(`${path}/`)
  );
}

export async function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;

  if (isPublicPath(pathname)) {
    return NextResponse.next();
  }

  const sessionToken = request.cookies.get("better-auth.session_token");

  if (!sessionToken?.value) {
    const loginUrl = new URL("/login", request.url);
    loginUrl.searchParams.set("callbackUrl", pathname);
    return NextResponse.redirect(loginUrl);
  }

  const response = NextResponse.next();

  response.headers.set("x-user-id", "authenticated");
  response.headers.set("x-authenticated", "true");

  return response;
}

export const config = {
  matcher: ["/((?!_next/static|_next/image|favicon.ico|fonts|images).*)"],
};
