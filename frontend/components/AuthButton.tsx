"use client";
import { useEffect, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL;

interface Me {
  id: string;
  email: string;
  name: string | null;
}

// Sign-in / signed-in affordance. `undefined` = still loading, `null` = signed out.
export function AuthButton() {
  const [me, setMe] = useState<Me | null | undefined>(undefined);

  useEffect(() => {
    fetch(`${API}/auth/me`, { credentials: "include" })
      .then((r) => (r.ok ? r.json() : null))
      .then(setMe)
      .catch(() => setMe(null));
  }, []);

  if (me === undefined) return <div className="h-5 w-16" />; // reserve space, no flicker

  if (!me) {
    const next = typeof window !== "undefined" ? window.location.pathname : "/";
    return (
      <a
        href={`${API}/auth/login?next=${encodeURIComponent(next)}`}
        className="text-sm text-amber-500 hover:text-amber-400 whitespace-nowrap"
      >
        Sign in
      </a>
    );
  }

  const signOut = async () => {
    await fetch(`${API}/auth/logout`, { method: "POST", credentials: "include" });
    window.location.reload();
  };

  return (
    <div className="flex items-center gap-3 text-sm">
      <span className="text-stone-400 truncate max-w-[14ch]" title={me.email}>
        {me.name || me.email}
      </span>
      <button onClick={signOut} className="text-stone-500 hover:text-stone-300 whitespace-nowrap">
        Sign out
      </button>
    </div>
  );
}
