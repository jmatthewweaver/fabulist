// Screen 1: Game List
import Link from "next/link";
import { AuthButton } from "../components/AuthButton";

async function getGames() {
  const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/games`, {
    cache: "no-store",
    credentials: "include",
  });
  if (!res.ok) return [];
  return res.json();
}

export default async function GameListPage() {
  const games = await getGames();

  return (
    <main className="max-w-2xl mx-auto px-4 py-8">
      <div className="flex items-start justify-between mb-8 gap-4">
        <div>
          <h1 className="text-3xl font-serif text-stone-100 mb-2">Fabulist</h1>
          <p className="text-stone-400 text-sm">AI-augmented interactive fiction</p>
        </div>
        <AuthButton />
      </div>

      {games.length === 0 ? (
        <p className="text-stone-500 text-sm">No games ingested yet. Add .z5 or .z8 files to the games/ directory and run ingestion.</p>
      ) : (
        <div className="grid gap-3">
          {games.map((game: any) => (
            <Link key={game.id} href={`/games/${game.id}`}>
              <div className="flex items-center gap-4 p-4 rounded-lg bg-stone-900 hover:bg-stone-800 transition-colors border border-stone-800 hover:border-stone-700">
                {game.icon_image_url ? (
                  <img src={game.icon_image_url} alt="" className="w-16 h-16 rounded object-cover flex-shrink-0" />
                ) : (
                  <div className="w-16 h-16 rounded image-placeholder flex-shrink-0" />
                )}
                <div>
                  <h2 className="font-medium text-stone-100">{game.title}</h2>
                  {game.description && (
                    <p className="text-stone-400 text-sm mt-1 line-clamp-2">{game.description}</p>
                  )}
                </div>
              </div>
            </Link>
          ))}
        </div>
      )}
    </main>
  );
}
