import "@testing-library/jest-dom/vitest";
import { vi } from "vitest";

process.env.DATABASE_URL = "postgresql://test:test@localhost:5432/test";
process.env.BETTER_AUTH_SECRET = "test-secret-min-32-chars-for-testing";
process.env.NEXT_PUBLIC_APP_URL = "http://localhost:3000";
process.env.RESEND_API_KEY = "re_test_key";

const mockDb = {
  select: vi.fn(() => ({
    from: vi.fn(() => ({
      where: vi.fn(() => Promise.resolve([])),
      innerJoin: vi.fn(() => ({
        where: vi.fn(() => Promise.resolve([])),
      })),
    })),
  })),
  insert: vi.fn(() => ({
    values: vi.fn(() => ({
      returning: vi.fn(() => Promise.resolve([{}])),
    })),
  })),
  update: vi.fn(() => ({
    set: vi.fn(() => ({
      where: vi.fn(() => Promise.resolve([])),
    })),
  })),
  delete: vi.fn(() => ({
    where: vi.fn(() => Promise.resolve([])),
  })),
};

vi.mock("@/lib/db", () => ({
  db: mockDb,
  users: {},
  sessions: {},
  accounts: {},
  verifications: {},
}));

vi.mock("resend", () => ({
  Resend: class {
    emails = {
      send: vi.fn().mockResolvedValue({ data: { id: "test-email-id" }, error: null }),
    };
  },
}));

vi.mock("@neondatabase/serverless", () => ({
  Pool: class {
    end = vi.fn();
  },
}));

vi.mock("drizzle-orm/neon-serverless", () => ({
  drizzle: vi.fn(() => mockDb),
}));

vi.mock("better-auth", () => ({
  betterAuth: vi.fn(() => ({
    api: {
      getSession: vi.fn(() => Promise.resolve(null)),
    },
  })),
}));

vi.mock("better-auth/adapters/drizzle", () => ({
  drizzleAdapter: vi.fn(() => ({})),
}));

vi.mock("better-auth/react", () => ({
  createAuthClient: vi.fn(() => ({
    signIn: {
      email: vi.fn(),
      social: vi.fn(),
    },
    signUp: {
      email: vi.fn(),
    },
    signOut: vi.fn(),
    useSession: vi.fn(() => ({ data: null, isPending: false })),
  })),
}));

Object.defineProperty(global, "crypto", {
  value: {
    randomUUID: vi.fn(() => "test-uuid-1234"),
    randomBytes: vi.fn((size: number) => ({
      toString: vi.fn(() => "a".repeat(size)),
    })),
    createHash: vi.fn(() => ({
      update: vi.fn().mockReturnThis(),
      digest: vi.fn(() => "test-hash-abcdef"),
    })),
  },
});

vi.mock("next/headers", () => ({
  headers: vi.fn(() => Promise.resolve(new Headers())),
}));

vi.mock("next/navigation", () => ({
  useRouter: vi.fn(() => ({
    push: vi.fn(),
    replace: vi.fn(),
    back: vi.fn(),
  })),
  useSearchParams: vi.fn(() => ({
    get: vi.fn(),
  })),
  usePathname: vi.fn(() => "/"),
}));
