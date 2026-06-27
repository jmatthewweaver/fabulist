// Screen 2: Game Page — description, default style, user's saves
"use client";
import { useEffect, useState } from "react";
import { useRouter, useParams } from "next/navigation";
import Link from "next/link";

interface Save {
  id: string;
  session_id: string;
  name: string;
  room_name: string;
  turn_count: number;
  created_at: string;
}

interface GameDetail {
  id: string;
  title: string;
  description: string;
  icon_image_url: string | null;
  default_style_id: string;
  saves: Save[];
  available_styles: { id: string; name: string; description: string }[];
}

export default function GamePage() {
  const { gameId } = useParams<{ gameId: string }>();
  const router = useRouter();
  const [game, setGame] = useState<GameDetail | null>(null);
  const [selectedStyle, setSelectedStyle] = useState<string>("");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [starting, setStarting] = useState(false);

  useEffect(() => {
    fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/games/${gameId}`, { credentials: "include" })
      .then((r) => r.json())
      .then((data) => {
        setGame(data);
        setSelectedStyle(data.default_style_id);
      });
  }, [gameId]);

  const startNew = async () => {
    setStarting(true);
    const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/sessions`, {
      method: "POST",
      credentials: "include",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ game_id: gameId, style_id: selectedStyle }),
    });
    const { session_id } = await res.json();
    router.push(`/play/${session_id}`);
  };

  const resume = (save: Save) => {
    router.push(`/play/${save.session_id}?restore=${save.id}`);
  };

  if (!game) return <div className="flex items-center justify-center h-screen text-stone-500">Loading...</div>;

  return (
    <main className="max-w-2xl mx-auto px-4 py-8">
      <Link href="/" className="text-stone-500 hover:text-stone-300 text-sm mb-6 block">← All games</Link>

      <div className="flex gap-5 mb-6">
        {game.icon_image_url ? (
          <img src={game.icon_image_url} alt="" className="w-24 h-24 rounded-lg object-cover flex-shrink-0" />
        ) : (
          <div className="w-24 h-24 rounded-lg image-placeholder flex-shrink-0" />
        )}
        <div>
          <h1 className="text-2xl font-serif text-stone-100">{game.title}</h1>
          {game.description && <p className="text-stone-400 text-sm mt-2">{game.description}</p>}
        </div>
      </div>

      {/* Saves */}
      {(game.saves ?? []).length > 0 && (
        <section className="mb-6">
          <h2 className="text-xs font-medium text-stone-500 uppercase tracking-wider mb-3">Continue</h2>
          <div className="grid gap-2">
            {(game.saves ?? []).map((save) => (
              <button
                key={save.id}
                onClick={() => resume(save)}
                className="flex items-center justify-between p-3 rounded-lg bg-stone-900 hover:bg-stone-800 border border-stone-800 text-left transition-colors"
              >
                <div>
                  <div className="text-sm text-stone-200">{save.name}</div>
                  <div className="text-xs text-stone-500 mt-0.5">{save.room_name} · Turn {save.turn_count}</div>
                </div>
                <span className="text-xs text-stone-600">{new Date(save.created_at).toLocaleDateString()}</span>
              </button>
            ))}
          </div>
        </section>
      )}

      {/* Start new */}
      <section>
        <button
          onClick={startNew}
          disabled={starting}
          className="w-full py-3 rounded-lg bg-amber-700 hover:bg-amber-600 text-stone-100 font-medium transition-colors disabled:opacity-50"
        >
          {starting ? "Starting..." : "Start New Game"}
        </button>

        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="mt-2 text-xs text-stone-600 hover:text-stone-400 w-full text-center"
        >
          {showAdvanced ? "Hide" : "Advanced options"}
        </button>

        {showAdvanced && (
          <div className="mt-3 p-4 rounded-lg bg-stone-900 border border-stone-800">
            <label className="block text-xs text-stone-400 mb-2">Visual style</label>
            <div className="grid gap-2">
              {game.available_styles.map((style) => (
                <label key={style.id} className="flex items-start gap-3 cursor-pointer">
                  <input
                    type="radio"
                    name="style"
                    value={style.id}
                    checked={selectedStyle === style.id}
                    onChange={() => setSelectedStyle(style.id)}
                    className="mt-0.5"
                  />
                  <div>
                    <div className="text-sm text-stone-200">{style.name}</div>
                    <div className="text-xs text-stone-500">{style.description}</div>
                  </div>
                </label>
              ))}
            </div>
          </div>
        )}
      </section>
    </main>
  );
}
